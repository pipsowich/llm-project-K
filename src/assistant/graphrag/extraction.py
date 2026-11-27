"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re
from collections.abc import Iterator

from pydantic import BaseModel, ValidationError

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

RELATION_MEANINGS = {
    "BELONGS_TO": "модель входит в линейку",
    "HAS_PART": "деталь установлена в модели",
    "IS_A": "деталь относится к этому виду узла",
    "INDICATES": "ошибка указывает на отказ узла",
    "FIXED_BY": "регламент устраняет ошибку",
    "REPLACED_BY": "устаревшая деталь заменена новой",
    "SUPPLIED_BY": "поставщик поставляет деталь",
    "FAILED": "отчёт о ремонте фиксирует отказ узла у модели",
}

INSTALLATION_LIST = re.compile(
    r"^В (?P<model>\S+) установлены следующие узлы\.[ \t]*\n[ \t\n]*(?P<items>(?:[ \t]*-[^\n]+\n?)+)",
    re.IGNORECASE | re.MULTILINE,
)
INSTALLED_PART = re.compile(
    r"^[ \t]*-[ \t]*(?P<component>[^\n]+?)[–—][ \t]*деталь[ \t]+(?P<part>[^,\s]+)"
    r"(?:,[ \t]*поставщик[ \t]*«(?P<supplier>[^»\n]+)»)?",
    re.IGNORECASE | re.MULTILINE,
)
PANEL_ERROR = re.compile(
    r"^[ \t]*-[ \t]*(?P<error>\S+)[ \t]*[–—][^\n]+?,[ \t]*отказ узла[ \t]*«(?P<component>[^»\n]+)»",
    re.IGNORECASE | re.MULTILINE,
)
EXPLICIT_REPLACEMENT = re.compile(
    r"\bдеталь\s+(?P<old>\S+)\s+(?:\([^\n]+\)\s+)?снята с производства\.\s+"
    r"вместо не[её] устанавливается деталь\s+(?P<new>[^\s.,;]+)",
    re.IGNORECASE,
)

EXAMPLE = {
    "entities": [
        {"type": "Part", "name": "P-1042"},
        {"type": "Part", "name": "P-1042B"},
        {"type": "Component", "name": "пусковое реле"},
    ],
    "relations": [
        {
            "subject_type": "Part",
            "subject": "P-1042",
            "type": "REPLACED_BY",
            "object_type": "Part",
            "object": "P-1042B",
        },
        {
            "subject_type": "Part",
            "subject": "P-1042",
            "type": "IS_A",
            "object_type": "Component",
            "object": "пусковое реле",
        },
    ],
}


class ExtractedEntity(BaseModel):
    type: str
    name: str


class ExtractedRelation(BaseModel):
    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str


class ExtractedFacts(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


def _document_relations(text: str, entities: set[Entity], names: dict[str, list[str]]) -> Iterator[Relation]:
    """Связывает найденные коды по явным строкам руководств и бюллетеней."""
    parts = {entity.name: entity for entity in entities if entity.type == "Part"}
    errors = {entity.name: entity for entity in entities if entity.type == "ErrorCode"}
    for scope in INSTALLATION_LIST.finditer(text):
        model = canonical_name("Model", scope["model"], {})
        if model is None:
            continue
        for row in INSTALLED_PART.finditer(scope["items"]):
            part = parts.get(canonical_name("Part", row["part"], {}))
            if part is None:
                continue
            yield Relation(Entity("Model", model), "HAS_PART", part)
            component = canonical_name("Component", row["component"], names)
            supplier = canonical_name("Supplier", row["supplier"] or "", names)
            if component is not None:
                yield Relation(part, "IS_A", Entity("Component", component))
            if supplier is not None:
                yield Relation(part, "SUPPLIED_BY", Entity("Supplier", supplier))

    for row in PANEL_ERROR.finditer(text):
        error = errors.get(canonical_name("ErrorCode", row["error"], {}))
        component = canonical_name("Component", row["component"], names)
        if error is not None and component is not None:
            yield Relation(error, "INDICATES", Entity("Component", component))

    for statement in EXPLICIT_REPLACEMENT.finditer(text):
        old = parts.get(canonical_name("Part", statement["old"], {}))
        new = parts.get(canonical_name("Part", statement["new"], {}))
        if old is not None and new is not None and old != new:
            yield Relation(old, "REPLACED_BY", new)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Извлекает факты и проверяет коды, названия и типы связей."""
    names = vocabulary()
    codes = set(find_codes(chunk.text))
    directions = "\n".join(
        f"{subject} -{relation}-> {object_type} ({RELATION_MEANINGS[relation]})"
        for relation, (subject, object_type) in RELATION_SIGNATURES.items()
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Извлеки явно описанные сущности и связи из фрагмента документа. "
                f"Типы сущностей: {', '.join(ENTITY_TYPES)}.\n"
                f"Допустимые названия: {json.dumps(names, ensure_ascii=False)}.\n"
                f"Связи направлены от subject к object:\n{directions}\n"
                "Коды в сущностях и связях копируй точно из списка кодов для фрагмента. "
                "Верни компактный JSON с массивами entities и relations. "
                "Если фактов нет, верни пустые массивы.\n"
                "Пример для текста «Деталь P-1042 (пусковое реле) заменена на P-1042B»: "
                + json.dumps(EXAMPLE, ensure_ascii=False, separators=(",", ":"))
            ),
        },
        {"role": "user", "content": f"Коды: {', '.join(sorted(codes)) or 'нет'}\nФрагмент:\n{chunk.text}"},
    ]
    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    has_structured_rows = INSTALLATION_LIST.search(chunk.text) or PANEL_ERROR.search(chunk.text)
    max_tokens = 2048
    if has_structured_rows:
        max_tokens = 1024
        messages[0]["content"] += (
            "\nСтроки списка установленных узлов или списка кодов ошибок обрабатываются отдельно. "
            "Из этих строк перечисли все сущности, особенно коды деталей и ошибок. "
            "Связи HAS_PART, IS_A, SUPPLIED_BY и INDICATES из самих списков в relations не повторяй. "
            "Связи вне списков, например принадлежность модели к линейке, извлеки по общей схеме."
        )

    for attempt in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=max_tokens)
        try:
            facts = ExtractedFacts.model_validate_json(response.content, strict=True)
        except ValidationError:
            if attempt == 0:
                messages = [
                    *messages,
                    {"role": "user", "content": "Повтори извлечение. Верни только корректный JSON по схеме."},
                ]
        else:
            break
    else:
        return result

    def grounded_entity(kind: str, raw_name: str) -> Entity | None:
        name = canonical_name(kind, raw_name, names)
        if name is None or (kind in CODE_TYPES and name not in codes):
            return None
        return Entity(kind, name)

    entities = set()
    relations = set()

    def keep_relation(relation: Relation) -> None:
        if relation in relations:
            return
        relations.add(relation)
        result.relations.append(relation)
        for entity in (relation.subject, relation.object):
            if entity not in entities:
                entities.add(entity)
                result.entities.append(entity)

    for raw in facts.entities:
        entity = grounded_entity(raw.type, raw.name)
        if entity is None:
            result.dropped += 1
        elif entity not in entities:
            entities.add(entity)
            result.entities.append(entity)

    for raw in facts.relations:
        subject = grounded_entity(raw.subject_type, raw.subject)
        target = grounded_entity(raw.object_type, raw.object)
        if subject is None or target is None:
            result.dropped += 1
            continue
        relation = Relation(subject, raw.type, target)
        if not relation.matches_schema():
            result.dropped += 1
        else:
            keep_relation(relation)

    for relation in _document_relations(chunk.text, entities, names):
        keep_relation(relation)
    return result
