"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет факты и считает вес каждого ребра по уникальным чанкам."""
    graph = nx.DiGraph()
    for result in results:
        entities = dict.fromkeys(result.entities)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            source, target = relation.subject.node_id, relation.object.node_id
            entities[relation.subject] = None
            entities[relation.object] = None
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, weight=0, sources=[])
            edge = graph[source][target]
            if result.chunk_id not in edge["sources"]:
                edge["sources"].append(result.chunk_id)
                edge["weight"] += 1

        for entity in entities:
            node = graph.nodes.get(entity.node_id)
            if node is None or "sources" not in node:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
            sources = graph.nodes[entity.node_id]["sources"]
            if result.chunk_id not in sources:
                sources.append(result.chunk_id)

    conflicts = 0
    for source, target in list(graph.edges):
        if source >= target:
            continue
        forward = graph[source][target]
        backward = graph.get_edge_data(target, source)
        if forward["type"] != "REPLACED_BY" or backward is None or backward["type"] != "REPLACED_BY":
            continue
        conflicts += 1
        if forward["weight"] <= backward["weight"]:
            graph.remove_edge(source, target)
        if backward["weight"] <= forward["weight"]:
            graph.remove_edge(target, source)

    graph.graph["conflicts"] = conflicts
    return graph
