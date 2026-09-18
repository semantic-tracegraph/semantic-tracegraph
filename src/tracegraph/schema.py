from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    REASONING = "reasoning"
    COMMAND = "command"
    OBSERVATION = "observation"
    SEARCH = "search"
    READ = "read"
    EDIT = "edit"
    TEST = "test"
    SUBMISSION = "submission"


class NodeType(StrEnum):
    LOCALIZATION = "localization"
    DIAGNOSIS = "diagnosis"
    REPRODUCTION = "reproduction"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    HANDOFF_ARTIFACT = "handoff_artifact"


class NodeStatus(StrEnum):
    CLAIMED = "claimed"
    EVIDENCED = "evidenced"
    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


class EdgeType(StrEnum):
    REQUIRES = "requires"
    PRODUCES = "produces"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    INVALIDATES = "invalidates"


@dataclass(slots=True)
class TraceEvent:
    event_id: str
    turn: int
    type: EventType
    content: str
    command: str | None = None
    returncode: int | None = None
    paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trace:
    instance_id: str
    model: str
    problem_statement: str
    events: list[TraceEvent]
    resolved: bool | None = None
    instance_cost: float | None = None
    api_calls: int | None = None
    split: str | None = None
    source_split: str | None = None


@dataclass(slots=True)
class Evidence:
    event_id: str
    kind: str
    excerpt: str
    mechanically_checked: bool = False


@dataclass(slots=True)
class AccomplishmentNode:
    node_id: str
    claim: str
    type: str
    event_span: tuple[str, str]
    evidence: list[Evidence] = field(default_factory=list)
    artifact_delta: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.CLAIMED
    confidence: float = 0.0
    remaining_work: str = ""
    semantic_key: str = ""
    weight: float = 1.0
    segment_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    type: str


@dataclass(slots=True)
class NodeTypeSpec:
    name: str
    description: str
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    evidence_requirements: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    default_weight: float = 1.0


@dataclass(slots=True)
class EdgeTypeSpec:
    name: str
    description: str
    is_dependency: bool = True
    source_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphSpec:
    spec_id: str
    version: str
    node_types: list[NodeTypeSpec]
    edge_types: list[EdgeTypeSpec]
    require_dag: bool = True
    allow_disconnected_roots: bool = True
    require_edges_if_multiple_nodes: bool = True
    typical_order: list[str] = field(default_factory=list)
    notes: str = ""

    def node_names(self) -> list[str]:
        return [item.name for item in self.node_types]

    def edge_names(self) -> list[str]:
        return [item.name for item in self.edge_types]

    def node_name_set(self) -> set[str]:
        return {item.name for item in self.node_types}

    def edge_name_set(self) -> set[str]:
        return {item.name for item in self.edge_types}

    def node_spec(self, name: str) -> NodeTypeSpec | None:
        for item in self.node_types:
            if item.name == name:
                return item
        return None

    def edge_spec(self, name: str) -> EdgeTypeSpec | None:
        for item in self.edge_types:
            if item.name == name:
                return item
        return None

    def dependency_edge_names(self) -> set[str]:
        return {item.name for item in self.edge_types if item.is_dependency}

    def weight_for(self, node_type: str) -> float:
        spec = self.node_spec(node_type)
        return spec.default_weight if spec is not None else 1.0


@dataclass(slots=True)
class AccomplishmentGraph:
    instance_id: str
    model: str
    nodes: list[AccomplishmentNode]
    edges: list[GraphEdge] = field(default_factory=list)
    resolved: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _typed_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in raw.items() if key in allowed}


def graph_spec_from_dict(data: dict[str, Any]) -> GraphSpec:
    payload = dict(data.get("spec") if "spec" in data and "node_types" not in data else data)
    payload["node_types"] = [
        NodeTypeSpec(**_typed_fields(NodeTypeSpec, dict(item)))
        for item in payload.get("node_types") or []
    ]
    payload["edge_types"] = [
        EdgeTypeSpec(**_typed_fields(EdgeTypeSpec, dict(item)))
        for item in payload.get("edge_types") or []
    ]
    return GraphSpec(**_typed_fields(GraphSpec, payload))


def graph_from_dict(data: dict[str, Any]) -> AccomplishmentGraph:
    nodes = []
    for raw in data.get("nodes", []):
        raw = dict(raw)
        raw["type"] = str(raw["type"])
        raw["status"] = NodeStatus(raw.get("status", "claimed"))
        raw["event_span"] = tuple(raw["event_span"])
        raw["evidence"] = [Evidence(**item) for item in raw.get("evidence", [])]
        nodes.append(AccomplishmentNode(**_typed_fields(AccomplishmentNode, raw)))
    edges = [
        GraphEdge(source=e["source"], target=e["target"], type=str(e["type"]))
        for e in data.get("edges", [])
    ]
    return AccomplishmentGraph(
        instance_id=data["instance_id"],
        model=data["model"],
        nodes=nodes,
        edges=edges,
        resolved=data.get("resolved"),
        metadata=data.get("metadata", {}),
    )


def trace_from_dict(data: dict[str, Any]) -> Trace:
    events = []
    for raw in data.get("events", []):
        raw = dict(raw)
        raw["type"] = EventType(raw["type"])
        events.append(TraceEvent(**raw))
    return Trace(
        instance_id=data["instance_id"],
        model=data["model"],
        problem_statement=data.get("problem_statement", ""),
        events=events,
        resolved=data.get("resolved"),
        instance_cost=data.get("instance_cost"),
        api_calls=data.get("api_calls"),
        split=data.get("split"),
        source_split=data.get("source_split"),
    )
