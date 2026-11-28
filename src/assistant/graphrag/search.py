"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from itertools import combinations, pairwise

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

NO_DATA = "В базе знаний не найдено."
FACT_LIMIT = 8000


class PartialAnswer(BaseModel):
    answer: str
    score: int = Field(ge=0, le=10)


def _document_ids(chunks: list[str]) -> set[str]:
    return {chunk.split("#", 1)[0] for chunk in chunks}


def _oriented_edges(graph: nx.DiGraph, left: str, right: str) -> list[tuple[str, str]]:
    return [(source, target) for source, target in ((left, right), (right, left)) if graph.has_edge(source, target)]


def _local_edges(kb: KnowledgeBase, anchors: list[str], question: str) -> dict[tuple[str, str], None]:
    graph = kb.graph
    undirected = graph.to_undirected(as_view=True)
    edges: dict[tuple[str, str], None] = {}
    if len(anchors) == 1:
        distances = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        for source, distance in sorted(distances.items(), key=lambda item: (item[1], item[0])):
            if distance < 2:
                for target in sorted(undirected[source]):
                    edges.update(dict.fromkeys(_oriented_edges(graph, source, target)))
        return edges

    for left, right in combinations(anchors, 2):
        paths = sorted(nx.all_simple_paths(undirected, left, right, cutoff=3), key=lambda path: (len(path), path))
        for path in paths:
            for source, target in pairwise(path):
                edges.update(dict.fromkeys(_oriented_edges(graph, source, target)))

    parts = {node for edge in edges for node in edge if graph.nodes[node]["type"] == "Part"}
    pending = sorted(parts)
    while pending:
        part = pending.pop()
        replacements = []
        for target, attrs in sorted(graph[part].items()):
            if attrs["type"] == "REPLACED_BY":
                edges[part, target] = None
                replacements.append(target)
                if target not in parts:
                    parts.add(target)
                    pending.append(target)
        if not replacements and "постав" in question.lower():
            for target, attrs in sorted(graph[part].items()):
                if attrs["type"] == "SUPPLIED_BY":
                    edges[part, target] = None
    return edges


def _answer(text: str, sources: set[str], facts: list[str] | None = None) -> Answer:
    text = text.strip()
    if not text or "в базе знаний не найдено" in text.lower():
        return Answer(NO_DATA, sources=sorted(sources), facts=facts or [], refused=True)
    return Answer(text, sources=sorted(sources), facts=facts or [])


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Передаёт модели факты на коротких путях или в окрестности одного узла."""
    graph = kb.graph
    code_nodes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in graph for node in code_nodes):
        return Answer(NO_DATA, refused=True)

    name_nodes = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    anchors = list(dict.fromkeys(code_nodes + [node for node in name_nodes if node in graph]))
    if not anchors:
        return Answer(NO_DATA, refused=True)

    facts, sources = [], set()
    length = 0
    for source, target in _local_edges(kb, anchors, question):
        attrs = graph[source][target]
        fact = f"{graph.nodes[source]['name']} -{attrs['type']}-> {graph.nodes[target]['name']}"
        line = f"{len(facts) + 1}. {fact}"
        length += len(line) + bool(facts)
        if length > FACT_LIMIT:
            break
        facts.append(line)
        sources.update(_document_ids(attrs.get("sources", [])))
    if not facts:
        return Answer(NO_DATA, refused=True)

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Ответь по списку фактов. INDICATES связывает ошибку с видом узла, "
                    "HAS_PART связывает модель с деталью, IS_A связывает деталь с видом узла. "
                    "Связывай именно запрошенную модель и нужный вид узла. "
                    "Цепочка REPLACED_BY направлена от старой детали к новой: "
                    "для заказа, замены и актуального поставщика используй последнюю деталь цепочки. "
                    f"Если данных для ответа нет, напиши «{NO_DATA}»"
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\nФакты:\n" + "\n".join(facts)},
        ]
    )
    return _answer(response.content, sources, facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Собирает частичные ответы по сводкам и объединяет полезные."""
    partials = []
    graph = kb.graph
    sources: set[str] = set()
    for community in kb.communities:
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Дай частичный ответ только по сводке, сохрани точные коды, числа, "
                        "модели и названия узлов. Оцени полезность от 0 до 10. "
                        "Поставь 0, если в сводке нет данных для этого вопроса."
                    ),
                },
                {"role": "user", "content": f"Вопрос: {question}\nСводка:\n{community.summary}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            partial = PartialAnswer.model_validate_json(response.content, strict=True)
        except ValidationError:
            continue
        if not partial.score or not partial.answer.strip():
            continue
        partials.append((partial.score, community.community_id, partial.answer))
        for node in community.nodes:
            sources.update(_document_ids(graph.nodes[node].get("sources", [])))
        for _, _, attrs in graph.out_edges(community.nodes, data=True):
            sources.update(_document_ids(attrs.get("sources", [])))

    if not partials:
        return Answer(NO_DATA, refused=True)
    context = "\n".join(
        f"Сообщество {number}: {text}" for _, number, text in sorted(partials, key=lambda item: -item[0])
    )
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы на вопрос. Сопоставь данные по запрошенной линейке. "
                    "Если приведены отдельные частоты отказов одного узла, сложи их по сообществам. "
                    "Сохраняй точные коды деталей и моделей. Используй только переданные сведения. "
                    f"Если данных для ответа нет, напиши «{NO_DATA}»"
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\nЧастичные ответы:\n{context}"},
        ]
    )
    return _answer(response.content, sources)
