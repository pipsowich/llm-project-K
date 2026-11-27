"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter, defaultdict

import networkx as nx

from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества Лувена с учётом веса связей."""
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph))
    for source, target, attrs in sorted(graph.edges(data=True)):
        weight = undirected.get_edge_data(source, target, {}).get("weight", 0)
        undirected.add_edge(source, target, weight=weight + attrs.get("weight", 1))

    if undirected.size(weight="weight"):
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    else:
        groups = [{node} for node in undirected]
    return [Community(number, nodes) for number, nodes in enumerate(sorted(sorted(group) for group in groups))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Формулирует сводку по фактам и рассчитанной статистике отказов."""
    failures_by_line: dict[str, Counter] = defaultdict(Counter)
    failures_by_model = Counter()
    retired_by_line: dict[str, set[str]] = defaultdict(set)
    facts = []
    for source, target, attrs in sorted(graph.out_edges(community.nodes, data=True)):
        relation = attrs["type"]
        weight = attrs.get("weight", 1)
        facts.append(f"{source} -{relation}-> {target} [вес {weight}]")
        if relation not in {"FAILED", "HAS_PART"}:
            continue
        lines = [node for node, edge in graph[source].items() if edge["type"] == "BELONGS_TO"]
        if relation == "FAILED":
            failures_by_model[source] += weight
            for line in lines:
                failures_by_line[line][target] += weight
        elif any(edge["type"] == "REPLACED_BY" for edge in graph[target].values()):
            for line in lines:
                retired_by_line[line].add(target)

    details = [
        "Узлы: " + ", ".join(community.nodes),
        "Отказы по линейкам и узлам: " + str({line: dict(counts) for line, counts in failures_by_line.items()}),
        "Отказы по моделям: " + str(dict(failures_by_model)),
        "Снятые детали из руководств по линейкам: "
        + str({line: sorted(parts) for line, parts in retired_by_line.items()}),
        "Связи:",
        *facts,
    ]
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши русскую сводку сообщества графа до 150 слов. "
                    "Используй только переданные факты и готовые числа. "
                    "В первую очередь сохрани названия линеек, коды моделей, частоты отказов "
                    "по моделям и узлам, а также коды снятых деталей из руководств. "
                    "Частоты описывают только это сообщество, не весь корпус. "
                    "По возможности перечисли существенные связи для обзорных вопросов."
                ),
            },
            {"role": "user", "content": "\n".join(details)},
        ]
    )
    summary = response.content.strip()
    words = summary.split()
    return " ".join(words[:150]) if len(words) > 150 else summary
