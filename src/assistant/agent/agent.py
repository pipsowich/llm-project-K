"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from decimal import Decimal

from assistant.agent.tools import ERP_NOT_FOUND, TOOL_SCHEMAS, validate_arguments
from assistant.graphrag.search import NO_DATA
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

ORDER = "create_service_order"
ERP_UNAVAILABLE = "Данные ERP получить не удалось."
INTEGER = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)"
NUMBER = rf"(?<![\w−-])(?<!\d[.,]){INTEGER}(?![\w-]|[.,]\d)"
AMOUNT = rf"(?<![\w−-])(?<!\d[.,])[+−-]?{INTEGER}(?:[.,]\d+)?(?![\w-]|[.,]\d)"
PRICE = r"(?:руб\w*|₽|RUB)"
UNITS = r"(?:шт\.?|штук\w*|ед\.?|единиц\w*|экземпляр\w*)"
PRICE_SUFFIX = re.compile(rf"({AMOUNT})\s*{PRICE}(?!\w)", re.IGNORECASE)
PRICE_PREFIX = re.compile(rf"(?:\b(?:цена|стоимост[ьи]|стоит)\s*[:–—=]?|₽)\s*({AMOUNT})", re.IGNORECASE)
STOCK_SUFFIX = re.compile(rf"({AMOUNT})\s*{UNITS}(?!\w)", re.IGNORECASE)
STOCK_PREFIX = re.compile(rf"\b(?:остаток|в наличии|на складе)\s*[:–—=]?\s*({AMOUNT})", re.IGNORECASE)
ORDER_ID = re.compile(r"(?<![\w-])SO-\d+(?![\w-])", re.IGNORECASE)
ORDER_NUMBERS = {
    "одну": 1,
    "одна": 1,
    "один": 1,
    "две": 2,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}
QUANTITY_TOKEN = rf"(?:{NUMBER}|\b(?:{'|'.join(ORDER_NUMBERS)})\b)"
ORDER_QUANTITY = re.compile(
    rf"({QUANTITY_TOKEN})\s*{UNITS}(?!\w)|"
    rf"\b(?:закажи(?:те)?|оформи(?:те)?(?:\s+заказ)?|заказ(?:ать)?|количество[м]?)"
    rf"\s*[:,]?\s*({QUANTITY_TOKEN})",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "Ты ассистент диспетчера «ПолярТех». Для вопросов о модели, ошибке и нужной детали используй "
    "search_knowledge, для обзора и сравнений по всему корпусу — summarize_topic. "
    "При запросе остатка или цены сначала установи код актуальной детали, затем вызови ERP. "
    "Для склада используй его название из схемы. Если ERP сообщает, что данных нет, так и скажи. "
    "Нулевой остаток означает, что данных достаточно и на складе 0 шт. "
    "Вызывай create_service_order только когда пользователь явно просит оформить заказ. "
    "Используй количество из вопроса. При отсутствии количества попроси уточнить. "
    "Используй только подтверждённые коды, цены и остатки. При недоступности ERP сообщи об этом. "
    f"Если в базе знаний нет сведений по запросу, ответь «{NO_DATA}»"
)


def _numeric_value(text: str) -> int:
    return int(re.sub(r"\s", "", text))


def _numbers(text: str, *patterns: re.Pattern) -> set[Decimal]:
    return {
        Decimal(re.sub(r"\s", "", match).replace(",", ".").replace("−", "-"))
        for pattern in patterns
        for match in pattern.findall(text)
    }


def _order_quantities(question: str) -> set[int]:
    quantities = set()
    for with_unit, after_verb in ORDER_QUANTITY.findall(question.lower()):
        token = with_unit or after_verb
        quantities.add(ORDER_NUMBERS[token] if token in ORDER_NUMBERS else _numeric_value(token))
    return quantities


def _call_with_retries(tool: Callable[..., dict], args: dict) -> dict:
    for _ in range(3):
        try:
            return tool(**args)
        except ErpUnavailable:
            continue
    return {"error": ERP_UNAVAILABLE}


def _record_result(step: ToolStep, result: dict) -> None:
    if "error" in result:
        step.error = result["error"]
    else:
        step.result = result


def _is_grounded(answer: str, question: str, steps: list[ToolStep]) -> bool:
    codes = set(find_codes(question))
    prices = _numbers(question, PRICE_PREFIX, PRICE_SUFFIX)
    quantities = _numbers(question, STOCK_PREFIX, STOCK_SUFFIX) | _order_quantities(question)
    orders = {order.upper() for order in ORDER_ID.findall(question)}
    for step in steps:
        if step.result is None:
            continue
        result = step.result
        evidence = result.get("facts") or [result.get("answer", "")]
        codes.update(find_codes(" ".join([*evidence, result.get("part_id", ""), result.get("model", "")])))
        if "price_rub" in result:
            prices.add(result["price_rub"])
        if "qty" in result:
            quantities.add(result["qty"])
        if "order_id" in result:
            orders.add(result["order_id"].upper())

    return (
        set(find_codes(answer)) <= codes
        and _numbers(answer, PRICE_PREFIX, PRICE_SUFFIX) <= prices
        and _numbers(answer, STOCK_PREFIX, STOCK_SUFFIX) <= quantities
        and {order.upper() for order in ORDER_ID.findall(answer)} <= orders
    )


def _finish(answer: str, steps: list[ToolStep], sources: set[str], *, refused: bool = False) -> AgentResult:
    for step in steps:
        if step.tool == ORDER and step.result and (order := step.result.get("order_id")):
            if order not in answer:
                answer += f" Заказ {order} создан."
    if any(step.error == ERP_UNAVAILABLE for step in steps) and ERP_UNAVAILABLE not in answer:
        answer += f" {ERP_UNAVAILABLE}"
    return AgentResult(answer.strip(), steps, sorted(sources), refused)


def _limit_answer(steps: list[ToolStep]) -> str:
    facts = []
    for step in steps:
        if step.result is None:
            if step.error and step.error.startswith(ERP_NOT_FOUND):
                facts.append(step.error)
            continue
        result = step.result
        if result.get("answer") and not result.get("refused"):
            facts.append(result["answer"])
        if step.tool == "check_stock":
            facts.append(f"{result['part_id']}, склад {result['warehouse']}: {result['qty']} шт.")
        if step.tool == "get_price":
            facts.append(f"{result['part_id']}: {result['price_rub']} руб.")
    facts = list(dict.fromkeys(facts))
    return " ".join([*facts, "Работа прервана: исчерпан лимит вызовов инструментов."])


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Выполняет инструменты с проверкой аргументов и фактов ответа."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    known_parts = {code for code in find_codes(question) if code_type(code) == "Part"}
    cache: dict[tuple[str, str], dict] = {}

    while len(steps) < max_steps:
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(reply.as_message())
        if not reply.tool_calls:
            text = reply.content.strip()
            if not _is_grounded(text, question, steps):
                messages.append(
                    {"role": "user", "content": "Исправь ответ. Используй только подтверждённые коды, цены и остатки."}
                )
                text = llm.chat(messages).content.strip()
                if not _is_grounded(text, question, steps):
                    return _finish(NO_DATA, steps, sources, refused=True)
            has_data = any(step.result is not None and not step.result.get("refused") for step in steps)
            no_data = any(
                (step.result and step.result.get("refused")) or (step.error or "").startswith(ERP_NOT_FOUND)
                for step in steps
            )
            if (no_data and not has_data) or not text or "в базе знаний не найдено" in text.lower():
                if any((step.error or "").startswith(ERP_NOT_FOUND) for step in steps):
                    return _finish(f"{NO_DATA} {ERP_NOT_FOUND}.", steps, sources, refused=True)
                return _finish(NO_DATA, steps, sources, refused=True)
            return _finish(text, steps, sources)

        for call in reply.tool_calls:
            if len(steps) >= max_steps:
                break
            args = dict(call.arguments)
            step = ToolStep(call.name, args)
            steps.append(step)

            if call.name not in tools or call.name not in _TOOL_NAMES:
                error = f"Неизвестный инструмент: {call.name}."
            else:
                error = validate_arguments(call.name, args)
            if not error and "part_id" in args:
                args["part_id"] = normalize_code(args["part_id"])
                if args["part_id"] not in known_parts:
                    error = "Код детали пока не подтверждён. Сначала вызови search_knowledge."
            if not error and call.name == ORDER:
                args["model"] = normalize_code(args["model"])
                if _order_quantities(question) != {args["qty"]}:
                    step.error = "Количество заказа требует уточнения."
                    return _finish("Сколько штук нужно заказать?", steps, sources)
                if not confirm:
                    step.error = "Требуется подтверждение заказа."
                    result = _finish(
                        f"Подтвердите заказ {args['part_id']} для {args['model']}: "
                        f"{args['qty']} шт., склад {args['warehouse']}.",
                        steps,
                        sources,
                    )
                    result.needs_confirmation = True
                    result.pending_action = {"tool": ORDER, "args": args}
                    return result

            if error:
                outcome = {"error": error}
            else:
                key = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if key not in cache:
                    cache[key] = _call_with_retries(tools[call.name], args)
                outcome = cache[key]
            _record_result(step, outcome)
            sources.update(outcome.get("sources", []))
            if not step.error and not outcome.get("refused"):
                evidence = outcome.get("facts") or [outcome.get("answer", "")]
                known_parts.update(
                    code
                    for code in find_codes(" ".join([*evidence, outcome.get("part_id", "")]))
                    if code_type(code) == "Part"
                )
            messages.append(tool_message(call, outcome))

    answer = _limit_answer(steps)
    if not _is_grounded(answer, question, steps):
        answer = f"{NO_DATA} Работа прервана: исчерпан лимит вызовов инструментов."
    return _finish(answer, steps, sources, refused=answer.startswith(NO_DATA))


_TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
