"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import canonical_name
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
ERP_NOT_FOUND = "Данных в ERP нет"

QUESTION = {"type": "string", "description": "Полный вопрос с нужными кодами и названиями"}
PART_ID = {"type": "string", "description": "Код детали, например P-1042B"}
WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
QTY = {"type": "integer", "minimum": 1}
MODEL = {"type": "string", "description": "Код модели оборудования, например AR-210"}

_TOOL_DEFINITIONS = {
    "search_knowledge": ("Найти точный ответ по модели, ошибке, детали или узлу", {"question": QUESTION}),
    "summarize_topic": ("Ответить на обзорный вопрос по всему корпусу", {"question": QUESTION}),
    "check_stock": ("Проверить остаток детали на указанном складе", {"part_id": PART_ID, "warehouse": WAREHOUSE}),
    "get_price": ("Узнать цену детали в рублях", {"part_id": PART_ID}),
    "create_service_order": (
        "Создать заказ детали только по явному запросу на оформление",
        {"part_id": PART_ID, "warehouse": WAREHOUSE, "qty": QTY, "model": MODEL},
    ),
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }
    for name, (description, properties) in _TOOL_DEFINITIONS.items()
]


def validate_arguments(name: str, args: dict) -> str | None:
    """Проверяет набор и значения параметров до вызова инструмента."""
    _, parameters = _TOOL_DEFINITIONS[name]
    if args.keys() != parameters.keys():
        return f"Укажите параметры: {', '.join(parameters)}."
    for key, value in args.items():
        if key == "qty":
            if type(value) is not int or value < 1:
                return "Количество должно быть положительным целым числом."
        elif not isinstance(value, str) or not value.strip():
            return f"Параметр {key} должен быть непустой строкой."
    if "part_id" in args and canonical_name("Part", args["part_id"], {}) is None:
        return "Укажите код детали вида P-1042 или P-1042B."
    if "warehouse" in args and args["warehouse"] not in WAREHOUSES:
        return f"Укажите один из складов: {', '.join(WAREHOUSES)}."
    if "model" in args and canonical_name("Model", args["model"], {}) is None:
        return "Укажите код модели вида AR-210."
    return None


def _erp_result(name: str, request: Callable[..., dict], args: dict) -> dict:
    error = validate_arguments(name, args)
    if error:
        return {"error": error}

    args["part_id"] = canonical_name("Part", args["part_id"], {})
    if "model" in args:
        args["model"] = canonical_name("Model", args["model"], {})

    try:
        return request(**args)
    except ErpNotFound as exc:
        return {"error": f"{ERP_NOT_FOUND}: {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Связывает инструменты с поиском по графу и ERP."""

    def search_knowledge(question: str) -> dict:
        error = validate_arguments("search_knowledge", {"question": question})
        if error:
            return {"error": error}
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        error = validate_arguments("summarize_topic", {"question": question})
        if error:
            return {"error": error}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        return _erp_result("check_stock", erp.get_stock, {"part_id": part_id, "warehouse": warehouse})

    def get_price(part_id: str) -> dict:
        return _erp_result("get_price", erp.get_price, {"part_id": part_id})

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        args = {"part_id": part_id, "warehouse": warehouse, "qty": qty, "model": model}
        return _erp_result("create_service_order", erp.create_order, args)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
