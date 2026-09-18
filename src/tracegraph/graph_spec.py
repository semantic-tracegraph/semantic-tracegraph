from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .schema import (
    AccomplishmentGraph,
    EdgeType,
    EdgeTypeSpec,
    GraphSpec,
    NodeType,
    NodeTypeSpec,
    graph_spec_from_dict,
    to_dict,
)

SPEC_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
DEFAULT_NODE_WEIGHTS = {
    NodeType.LOCALIZATION.value: 0.6,
    NodeType.DIAGNOSIS.value: 0.9,
    NodeType.REPRODUCTION.value: 1.0,
    NodeType.IMPLEMENTATION.value: 1.2,
    NodeType.VERIFICATION.value: 1.2,
    NodeType.HANDOFF_ARTIFACT.value: 0.7,
}


DEFAULT_GRAPH_SPEC = GraphSpec(
    spec_id="default-coding-agent",
    version="1",
    notes=(
        "Default ontology for SWE-style coding-agent trajectories: coarse, "
        "non-overlapping accomplishments grounded in trace events."
    ),
    typical_order=[
        NodeType.LOCALIZATION.value,
        NodeType.DIAGNOSIS.value,
        NodeType.REPRODUCTION.value,
        NodeType.IMPLEMENTATION.value,
        NodeType.VERIFICATION.value,
        NodeType.HANDOFF_ARTIFACT.value,
    ],
    node_types=[
        NodeTypeSpec(
            name=NodeType.LOCALIZATION.value,
            description=(
                "Found the relevant code, files, symbols, or failure site. "
                "This is search and reading that identifies where the work lives."
            ),
            inclusion_criteria=[
                "The agent located a specific file, symbol, or region that later work depends on.",
                "Search, read, or grep evidence shows the relevant site was found.",
            ],
            exclusion_criteria=[
                "Do not emit a localization node for every search; merge related lookups.",
                "Do not use localization for a diagnosis of why the bug happens.",
            ],
            evidence_requirements=[
                "Cite the search/read events and the observation that names the site.",
            ],
            examples=["Located should_close in src/message.py via rg."],
            default_weight=DEFAULT_NODE_WEIGHTS[NodeType.LOCALIZATION.value],
        ),
        NodeTypeSpec(
            name=NodeType.DIAGNOSIS.value,
            description=(
                "Explained the causal mechanism of the bug or the required change. "
                "A diagnosis is a specific claim about why current behavior is wrong."
            ),
            inclusion_criteria=[
                "The agent stated a concrete cause, invariant, or missing condition.",
            ],
            exclusion_criteria=[
                "Do not label a bare code edit as diagnosis without a causal claim.",
            ],
            evidence_requirements=[
                "Cite reasoning or observations that support the causal claim.",
            ],
            examples=["Identified an early return that made later header logic unreachable."],
            default_weight=DEFAULT_NODE_WEIGHTS[NodeType.DIAGNOSIS.value],
        ),
        NodeTypeSpec(
            name=NodeType.REPRODUCTION.value,
            description=(
                "Reproduced the failing behavior, wrote a failing test, or otherwise "
                "demonstrated the bug before or while fixing it."
            ),
            inclusion_criteria=[
                "A command or test observed the failure, or a new failing test was added.",
            ],
            exclusion_criteria=[
                "Passing tests after a fix belong to verification, not reproduction.",
            ],
            evidence_requirements=[
                "Cite the command and the failing or reproducing observation.",
            ],
            examples=["Ran a failing unit test that showed the header bug."],
            default_weight=DEFAULT_NODE_WEIGHTS[NodeType.REPRODUCTION.value],
        ),
        NodeTypeSpec(
            name=NodeType.IMPLEMENTATION.value,
            description=(
                "Changed source to address the task: edits, new code, or refactors "
                "that implement the fix or feature."
            ),
            inclusion_criteria=[
                "The agent modified task-relevant files or generated a patch.",
            ],
            exclusion_criteria=[
                "Do not split every tiny edit into its own node; merge edits that serve one fix.",
            ],
            evidence_requirements=[
                "Cite edit events and record changed paths in artifact_delta.paths.",
            ],
            examples=["Patched message.py to move the version fallback after the header loop."],
            default_weight=DEFAULT_NODE_WEIGHTS[NodeType.IMPLEMENTATION.value],
        ),
        NodeTypeSpec(
            name=NodeType.VERIFICATION.value,
            description=(
                "Checked that the change works: tests, reproductions after the fix, "
                "or other mechanical confirmation."
            ),
            inclusion_criteria=[
                "Tests were run, or the agent otherwise checked the post-fix behavior.",
            ],
            exclusion_criteria=[
                "A successful edit returncode is not verification by itself.",
            ],
            evidence_requirements=[
                "Cite the test/check command and its observation.",
            ],
            examples=["Ran pytest and observed 1 passed."],
            default_weight=DEFAULT_NODE_WEIGHTS[NodeType.VERIFICATION.value],
        ),
        NodeTypeSpec(
            name=NodeType.HANDOFF_ARTIFACT.value,
            description=(
                "Produced the deliverable for the next agent or evaluator: a submitted "
                "patch, git diff, or explicit completion artifact."
            ),
            inclusion_criteria=[
                "The agent emitted a patch, committed diff, or final submission payload.",
            ],
            exclusion_criteria=[
                "Do not use handoff for intermediate edits that are not the delivered artifact.",
            ],
            evidence_requirements=[
                "Cite the submission command and the artifact observation.",
            ],
            examples=["Submitted git diff --cached of the message.py patch."],
            default_weight=DEFAULT_NODE_WEIGHTS[NodeType.HANDOFF_ARTIFACT.value],
        ),
    ],
    edge_types=[
        EdgeTypeSpec(
            name=EdgeType.REQUIRES.value,
            description="The target could not start until the source accomplishment was done.",
            is_dependency=True,
        ),
        EdgeTypeSpec(
            name=EdgeType.PRODUCES.value,
            description="The source yields an artifact or result that the target consumes.",
            is_dependency=True,
        ),
        EdgeTypeSpec(
            name=EdgeType.REFINES.value,
            description="The target improves, corrects, or narrows the source accomplishment.",
            is_dependency=True,
        ),
        EdgeTypeSpec(
            name=EdgeType.CONTRADICTS.value,
            description="The target conflicts with the source claim; it does not create a prerequisite.",
            is_dependency=False,
        ),
        EdgeTypeSpec(
            name=EdgeType.INVALIDATES.value,
            description="The target undoes or disproves the source; it does not create a prerequisite.",
            is_dependency=False,
        ),
    ],
)


def spec_hash(spec: GraphSpec) -> str:
    canonical = json.dumps(to_dict(spec), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def spec_metadata(spec: GraphSpec) -> dict[str, Any]:
    return {
        "graph_spec": to_dict(spec),
        "graph_spec_id": spec.spec_id,
        "graph_spec_hash": spec_hash(spec),
    }


def spec_from_graph(graph: AccomplishmentGraph) -> GraphSpec:
    raw = graph.metadata.get("graph_spec")
    if isinstance(raw, dict) and raw.get("node_types"):
        return graph_spec_from_dict(raw)
    return DEFAULT_GRAPH_SPEC


def load_graph_spec(path: str | Path) -> GraphSpec:
    data = json.loads(Path(path).read_text())
    spec = graph_spec_from_dict(data)
    errors = validate_graph_spec(spec)
    if errors:
        raise ValueError("Invalid graph spec: " + "; ".join(errors))
    return spec


def validate_graph_spec(spec: GraphSpec | dict[str, Any]) -> list[str]:
    if isinstance(spec, dict):
        try:
            spec = graph_spec_from_dict(spec)
        except (TypeError, KeyError, ValueError) as exc:
            return [f"spec could not be parsed: {exc}"]
    errors: list[str] = []
    if not str(spec.spec_id or "").strip():
        errors.append("spec_id is required.")
    if not str(spec.version or "").strip():
        errors.append("version is required.")
    if not spec.node_types:
        errors.append("spec must define at least one node type.")
    if not spec.edge_types:
        errors.append("spec must define at least one edge type.")

    seen_nodes: set[str] = set()
    for item in spec.node_types:
        errors.extend(_validate_type_name("node type", item.name, seen_nodes))
        if not str(item.description or "").strip():
            errors.append(f"Node type {item.name} needs a description.")
        if not isinstance(item.default_weight, (int, float)) or float(item.default_weight) <= 0:
            errors.append(f"Node type {item.name} default_weight must be a positive number.")

    seen_edges: set[str] = set()
    for item in spec.edge_types:
        errors.extend(_validate_type_name("edge type", item.name, seen_edges))
        if not str(item.description or "").strip():
            errors.append(f"Edge type {item.name} needs a description.")
        unknown_sources = [name for name in item.source_types if name not in seen_nodes]
        unknown_targets = [name for name in item.target_types if name not in seen_nodes]
        if unknown_sources:
            errors.append(f"Edge type {item.name} source_types not in node types: {unknown_sources}.")
        if unknown_targets:
            errors.append(f"Edge type {item.name} target_types not in node types: {unknown_targets}.")

    if spec.edge_types and not spec.dependency_edge_names():
        errors.append("At least one edge type must have is_dependency=true.")

    unknown_order = [name for name in spec.typical_order if name not in seen_nodes]
    if unknown_order:
        errors.append(f"typical_order contains unknown node types: {unknown_order}.")
    return errors


def render_graph_spec(spec: GraphSpec) -> str:
    node_blocks = []
    for item in spec.node_types:
        lines = [f"- `{item.name}` (weight {item.default_weight}): {item.description.strip()}"]
        for label, values in (
            ("Include", item.inclusion_criteria),
            ("Exclude", item.exclusion_criteria),
            ("Evidence", item.evidence_requirements),
            ("Examples", item.examples),
        ):
            if values:
                lines.append(f"  {label}: " + " ".join(str(value).strip() for value in values))
        node_blocks.append("\n".join(lines))

    edge_blocks = []
    for item in spec.edge_types:
        role = "dependency" if item.is_dependency else "non-dependency"
        constraints = []
        if item.source_types:
            constraints.append("sources=" + ",".join(item.source_types))
        if item.target_types:
            constraints.append("targets=" + ",".join(item.target_types))
        suffix = f" [{'; '.join(constraints)}]" if constraints else ""
        edge_blocks.append(f"- `{item.name}` ({role}): {item.description.strip()}{suffix}")

    order = " -> ".join(spec.typical_order) if spec.typical_order else "not specified"
    roots = (
        "A graph may have multiple root nodes when the trajectory contains genuinely "
        "independent starting accomplishments."
        if spec.allow_disconnected_roots
        else "The graph should have a single connected causal structure."
    )
    dag = (
        "The graph must be a DAG (no cycles)."
        if spec.require_dag
        else "Cycles are allowed only when the spec explicitly permits them."
    )
    notes = f"\n\nSpec notes: {spec.notes.strip()}" if spec.notes.strip() else ""
    return (
        f"Graph spec `{spec.spec_id}` v{spec.version}. Use only the node and edge types below.\n\n"
        f"Node types:\n" + "\n".join(node_blocks) + "\n\n"
        f"Edge types:\n" + "\n".join(edge_blocks) + "\n\n"
        f"Typical causal order: {order}.\n"
        f"{dag} {roots} Do not add an edge merely because two events happened in sequence."
        f"{notes}"
    )


def _validate_type_name(kind: str, name: str, seen: set[str]) -> list[str]:
    errors: list[str] = []
    if not name or not SPEC_NAME_RE.fullmatch(name):
        errors.append(
            f"Invalid {kind} name {name!r}; use snake_case starting with a letter "
            "(e.g. localization, handoff_artifact)."
        )
        return errors
    if name in seen:
        errors.append(f"Duplicate {kind} name: {name}")
    seen.add(name)
    return errors
