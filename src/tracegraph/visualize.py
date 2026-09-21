from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import AccomplishmentGraph, AccomplishmentNode, Trace, to_dict


EDGE_PRIORITY = {"requires": 0, "refines": 1, "produces": 2}
_EVENT_NUMBER = re.compile(r"(\d+)$")


def _event_position(event_id: str, positions: dict[str, int]) -> int:
    if event_id in positions:
        return positions[event_id]
    match = _EVENT_NUMBER.search(event_id)
    return int(match.group(1)) if match else 0


def _node_bounds(
    node: AccomplishmentNode,
    positions: dict[str, int],
) -> tuple[int, int]:
    start = _event_position(node.event_span[0], positions)
    end = _event_position(node.event_span[1], positions)
    return start, max(start, end)


def derive_primary_parents(
    graph: AccomplishmentGraph,
    trace: Trace | None = None,
) -> dict[str, str | None]:
    """Project a graph onto a tree by selecting one incoming dependency edge.

    Edge type is the first tie-breaker. Within one type, the closest earlier
    source is preferred so the projection follows the trajectory where possible.
    Nodes without a usable incoming dependency become children of the task root.
    """

    positions = (
        {event.event_id: index for index, event in enumerate(trace.events)}
        if trace is not None
        else {}
    )
    nodes = {node.node_id: node for node in graph.nodes}
    bounds = {node_id: _node_bounds(node, positions) for node_id, node in nodes.items()}
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in graph.edges:
        if (
            edge.type in EDGE_PRIORITY
            and edge.source in nodes
            and edge.target in nodes
            and edge.source != edge.target
        ):
            incoming[edge.target].append((edge.source, edge.type))

    parents: dict[str, str | None] = {}
    for node in graph.nodes:
        target_start = bounds[node.node_id][0]

        def rank(candidate: tuple[str, str]) -> tuple[int, int, int, str]:
            source, edge_type = candidate
            source_start, source_end = bounds[source]
            is_after_target = int(source_start > target_start)
            distance = abs(target_start - source_end)
            return EDGE_PRIORITY[edge_type], is_after_target, distance, source

        candidates = incoming.get(node.node_id, [])
        parents[node.node_id] = min(candidates, key=rank)[0] if candidates else None
    return parents


def graph_visualization_data(
    graph: AccomplishmentGraph,
    trace: Trace | None = None,
) -> dict[str, Any]:
    positions = (
        {event.event_id: index for index, event in enumerate(trace.events)}
        if trace is not None
        else {}
    )
    parents = derive_primary_parents(graph, trace)
    node_ids = {node.node_id for node in graph.nodes}
    selected_edges = {
        (parent, node_id)
        for node_id, parent in parents.items()
        if parent is not None
    }
    secondary_edges = [
        to_dict(edge)
        for edge in graph.edges
        if (edge.source, edge.target) not in selected_edges
    ]
    event_tokens = (
        {
            event.event_id: max(1, math.ceil(len(event.content or "") / 4))
            for event in trace.events
        }
        if trace is not None
        else {}
    )
    nodes = []
    for node in graph.nodes:
        start, end = _node_bounds(node, positions)
        if trace is not None:
            span_events = trace.events[start : end + 1]
            own_weight = sum(event_tokens[event.event_id] for event in span_events)
        else:
            own_weight = max(1, end - start + 1)
        raw = to_dict(node)
        raw.update(
            {
                "primary_parent_id": parents[node.node_id],
                "start_order": start,
                "end_order": end,
                "own_weight": max(1, own_weight),
            }
        )
        nodes.append(raw)
    nodes.sort(key=lambda item: (item["start_order"], item["end_order"], item["node_id"]))
    problem = trace.problem_statement if trace is not None else ""
    return {
        "instance_id": graph.instance_id,
        "model": graph.model,
        "resolved": graph.resolved,
        "problem_statement": problem,
        "weight_unit": "estimated_tokens" if trace is not None else "events",
        "weight_note": (
            "Estimated from transcript text at four characters per token; "
            "the Cursor cache does not include exact token usage."
            if trace is not None
            else "Event count is used because the input does not contain its source trace."
        ),
        "nodes": nodes,
        "edges": [to_dict(edge) for edge in graph.edges],
        "secondary_edges": secondary_edges,
    }


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_visualization_html(graphs: list[dict[str, Any]]) -> str:
    if not graphs:
        raise ValueError("At least one graph is required.")
    payload = _safe_json(graphs)
    return _HTML_TEMPLATE.replace("__TRACEGRAPH_DATA__", payload)


def write_visualization_html(graphs: list[dict[str, Any]], output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_visualization_html(graphs), encoding="utf-8")


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TraceGraph task decomposition</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111315;
      --panel: #191c1f;
      --panel-2: #202429;
      --line: #343a40;
      --text: #f1f3f5;
      --muted: #9aa1a9;
      --accent: #ffd166;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(17, 19, 21, .96);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .brand { font-size: 15px; font-weight: 750; letter-spacing: .01em; margin-right: auto; }
    select, button {
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 7px 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover { border-color: #69717a; }
    main { padding: 22px; }
    .title-row { display: flex; align-items: flex-start; gap: 20px; margin-bottom: 18px; }
    .title-copy { min-width: 0; flex: 1; }
    h1 { font-size: clamp(19px, 2.2vw, 28px); line-height: 1.22; margin: 0 0 7px; }
    .meta { color: var(--muted); }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 12px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12px; }
    .swatch { width: 10px; height: 10px; border-radius: 3px; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 16px; }
    .chart-panel, .detail-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      min-width: 0;
    }
    .chart-panel { padding: 12px; overflow-x: auto; }
    .chart-help { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin: 0 2px 10px; }
    #chart { min-width: 720px; }
    svg { display: block; width: 100%; }
    .bar { cursor: pointer; transition: filter .12s ease, opacity .12s ease; }
    .bar:hover { filter: brightness(1.16); }
    .bar.selected { stroke: white; stroke-width: 2; }
    .bar-label { pointer-events: none; fill: #101214; font-weight: 650; font-size: 12px; }
    .bar-metric { pointer-events: none; fill: #101214; font-size: 11px; font-variant-numeric: tabular-nums; }
    .bar-label.root, .bar-metric.root { fill: #fff; }
    .detail-panel { padding: 16px; align-self: start; position: sticky; top: 74px; }
    .detail-panel h2 { font-size: 17px; line-height: 1.3; margin: 0 0 8px; }
    .eyebrow { color: var(--accent); font-size: 11px; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 7px; }
    .detail-grid { display: grid; grid-template-columns: 82px 1fr; gap: 7px 10px; margin: 14px 0; }
    .detail-grid dt { color: var(--muted); }
    .detail-grid dd { margin: 0; overflow-wrap: anywhere; }
    .section { border-top: 1px solid var(--line); margin-top: 14px; padding-top: 13px; }
    .section h3 { font-size: 12px; color: var(--muted); margin: 0 0 8px; text-transform: uppercase; letter-spacing: .06em; }
    .section p { margin: 6px 0; overflow-wrap: anywhere; }
    .empty { color: var(--muted); }
    .crumbs { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin: 0 2px 10px; min-height: 27px; }
    .crumbs button { padding: 3px 7px; font-size: 12px; }
    .crumbs i { color: var(--muted); font-style: normal; }
    @media (max-width: 900px) {
      .workspace { grid-template-columns: 1fr; }
      .detail-panel { position: static; }
      .title-row { display: block; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">TraceGraph · Task decomposition</div>
    <select id="graphSelect" aria-label="Choose trajectory"></select>
    <button id="resetButton" type="button">Reset zoom</button>
    <button id="flipButton" type="button">Flip orientation</button>
  </header>
  <main>
    <div class="title-row">
      <div class="title-copy">
        <h1 id="taskTitle"></h1>
        <div class="meta" id="taskMeta"></div>
        <div class="legend" id="legend"></div>
      </div>
    </div>
    <div class="workspace">
      <section class="chart-panel">
        <div class="chart-help"><span>Click a task to inspect and zoom into its subtasks.</span><span id="widthHelp"></span></div>
        <div class="crumbs" id="crumbs"></div>
        <div id="chart"></div>
      </section>
      <aside class="detail-panel" id="details"></aside>
    </div>
  </main>
  <script>
    const GRAPHS = __TRACEGRAPH_DATA__;
    const COLORS = {
      localization: "#70d6ff", diagnosis: "#ff70a6", reproduction: "#ff9770",
      implementation: "#ffd670", verification: "#7ae582", handoff_artifact: "#b8a1ff",
      root: "#4b5563", other: "#a7b0ba"
    };
    const state = { graphIndex: 0, zoomId: "__root__", selectedId: "__root__", flipped: false };
    const select = document.getElementById("graphSelect");
    GRAPHS.forEach((graph, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${graph.instance_id} · ${graph.model}`;
      select.appendChild(option);
    });
    select.hidden = GRAPHS.length < 2;
    select.addEventListener("change", () => {
      state.graphIndex = Number(select.value); state.zoomId = "__root__"; state.selectedId = "__root__"; render();
    });
    document.getElementById("resetButton").addEventListener("click", () => {
      state.zoomId = "__root__"; render();
    });
    document.getElementById("flipButton").addEventListener("click", () => {
      state.flipped = !state.flipped; render();
    });

    function buildTree(graph) {
      const root = {
        node_id: "__root__", claim: graph.problem_statement || graph.instance_id,
        type: "root", status: "", primary_parent_id: null, own_weight: 0,
        event_span: [], evidence: [], artifact_delta: {}, remaining_work: ""
      };
      const byId = new Map([[root.node_id, root]]);
      graph.nodes.forEach(node => byId.set(node.node_id, {...node}));
      byId.forEach(node => node.children = []);
      graph.nodes.forEach(node => {
        const parent = byId.get(node.primary_parent_id) || root;
        parent.children.push(byId.get(node.node_id));
      });
      const order = (node) => {
        node.children.sort((a, b) => a.start_order - b.start_order || a.node_id.localeCompare(b.node_id));
        node.children.forEach(order);
        node.total_weight = Math.max(1, Number(node.own_weight || 0) + node.children.reduce((sum, child) => sum + child.total_weight, 0));
      };
      order(root);
      return {root, byId};
    }

    function descendants(node, depth = 0, rows = []) {
      rows.push({node, depth});
      node.children.forEach(child => descendants(child, depth + 1, rows));
      return rows;
    }

    function layoutTree(root) {
      const items = [];
      function visit(node, x, width, depth) {
        items.push({node, x, width, depth});
        let cursor = x;
        const childTotal = node.children.reduce((sum, child) => sum + child.total_weight, 0);
        const available = node.node_id === "__root__" ? width : width * Math.min(1, childTotal / node.total_weight);
        node.children.forEach(child => {
          const childWidth = childTotal ? available * child.total_weight / childTotal : 0;
          visit(child, cursor, childWidth, depth + 1);
          cursor += childWidth;
        });
      }
      visit(root, 0, 1000, 0);
      return items;
    }

    function truncate(text, width) {
      if (!text) return "";
      const max = Math.max(0, Math.floor(width / 7.2) - 2);
      if (max < 4) return "";
      return text.length > max ? text.slice(0, max - 1) + "…" : text;
    }

    function formatEffort(value, graph) {
      const rounded = Math.round(Number(value || 0));
      if (graph.weight_unit === "estimated_tokens") {
        const compact = rounded >= 1000 ? `${(rounded / 1000).toFixed(rounded >= 10000 ? 0 : 1)}k` : String(rounded);
        return `≈${compact} tok`;
      }
      return `${rounded} event${rounded === 1 ? "" : "s"}`;
    }

    function el(name, attrs = {}, text = "") {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      if (text) node.textContent = text;
      return node;
    }

    function renderChart(tree, graph) {
      const zoomNode = tree.byId.get(state.zoomId) || tree.root;
      const layout = layoutTree(zoomNode);
      const maxDepth = Math.max(...layout.map(item => item.depth));
      const rowHeight = 44, gap = 3, height = (maxDepth + 1) * rowHeight + 4;
      const svg = el("svg", {viewBox: `0 0 1000 ${height}`, height});
      layout.forEach(item => {
        const yDepth = state.flipped ? item.depth : maxDepth - item.depth;
        const y = yDepth * rowHeight + 2;
        const group = el("g");
        const rect = el("rect", {
          x: item.x + 1, y, width: Math.max(0, item.width - 2), height: rowHeight - gap,
          rx: 5, fill: COLORS[item.node.type] || COLORS.other,
          class: `bar${item.node.node_id === state.selectedId ? " selected" : ""}`,
          "data-id": item.node.node_id
        });
        rect.addEventListener("click", () => {
          state.selectedId = item.node.node_id;
          if (item.node.children.length) state.zoomId = item.node.node_id;
          render();
        });
        group.appendChild(rect);
        const metric = formatEffort(item.node.total_weight, graph);
        const metricSpace = item.width > 145 ? Math.min(92, metric.length * 7 + 14) : 0;
        const label = truncate(item.node.claim, item.width - metricSpace);
        if (label) group.appendChild(el("text", {
          x: item.x + 9, y: y + 25,
          class: `bar-label${item.node.type === "root" ? " root" : ""}`
        }, label));
        if (metricSpace) group.appendChild(el("text", {
          x: item.x + item.width - 9, y: y + 25, "text-anchor": "end",
          class: `bar-metric${item.node.type === "root" ? " root" : ""}`
        }, metric));
        const title = el("title", {}, `${item.node.claim}\n${item.node.type}${item.node.status ? " · " + item.node.status : ""}\n${metric} inclusive`);
        group.appendChild(title);
        svg.appendChild(group);
      });
      const chart = document.getElementById("chart");
      chart.replaceChildren(svg);
    }

    function escapeText(value) {
      const span = document.createElement("span"); span.textContent = value == null ? "" : String(value); return span.innerHTML;
    }

    function renderDetails(node, graph) {
      const panel = document.getElementById("details");
      if (node.node_id === "__root__") {
        panel.innerHTML = `<div class="eyebrow">Overall task</div><h2>${escapeText(node.claim)}</h2>` +
          `<dl class="detail-grid"><dt>Model</dt><dd>${escapeText(graph.model)}</dd>` +
          `<dt>Total effort</dt><dd>${escapeText(formatEffort(node.total_weight, graph))}</dd>` +
          `<dt>Nodes</dt><dd>${graph.nodes.length}</dd><dt>Edges</dt><dd>${graph.edges.length}</dd></dl>` +
          `<p class="empty">Select a bar to inspect that subtask.</p>`;
        return;
      }
      const parent = graph.nodes.find(item => item.node_id === node.primary_parent_id);
      const links = graph.edges.filter(edge => edge.source === node.node_id || edge.target === node.node_id);
      const evidence = (node.evidence || []).map(item =>
        `<p><strong>${escapeText(item.event_id)}</strong> · ${escapeText(item.kind)}<br>${escapeText(item.excerpt)}</p>`
      ).join("") || `<p class="empty">No evidence excerpts.</p>`;
      const linkText = links.map(edge =>
        `<p>${escapeText(edge.source)} <strong>—${escapeText(edge.type)}→</strong> ${escapeText(edge.target)}</p>`
      ).join("") || `<p class="empty">No graph edges.</p>`;
      const paths = ((node.artifact_delta || {}).paths || []).join(", ") || "—";
      panel.innerHTML = `<div class="eyebrow">${escapeText(node.type)}</div><h2>${escapeText(node.claim)}</h2>` +
        `<dl class="detail-grid"><dt>Status</dt><dd>${escapeText(node.status)}</dd>` +
        `<dt>Parent</dt><dd>${escapeText(parent ? parent.claim : "Overall task")}</dd>` +
        `<dt>Own effort</dt><dd>${escapeText(formatEffort(node.own_weight, graph))}</dd>` +
        `<dt>With subtasks</dt><dd>${escapeText(formatEffort(node.total_weight, graph))}</dd>` +
        `<dt>Span</dt><dd>${escapeText((node.event_span || []).join(" → "))}</dd>` +
        `<dt>Files</dt><dd>${escapeText(paths)}</dd></dl>` +
        `<div class="section"><h3>Evidence</h3>${evidence}</div>` +
        `<div class="section"><h3>Graph relationships</h3>${linkText}</div>` +
        (node.remaining_work ? `<div class="section"><h3>Remaining work</h3><p>${escapeText(node.remaining_work)}</p></div>` : "");
    }

    function renderCrumbs(tree) {
      const trail = [];
      let current = tree.byId.get(state.zoomId) || tree.root;
      while (current) {
        trail.unshift(current);
        current = current.primary_parent_id ? tree.byId.get(current.primary_parent_id) : null;
      }
      const holder = document.getElementById("crumbs"); holder.replaceChildren();
      trail.forEach((node, index) => {
        if (index) holder.appendChild(Object.assign(document.createElement("i"), {textContent: "›"}));
        const button = document.createElement("button");
        button.textContent = node.node_id === "__root__" ? "Overall task" : node.claim;
        button.addEventListener("click", () => { state.zoomId = node.node_id; render(); });
        holder.appendChild(button);
      });
    }

    function renderLegend(graph) {
      const types = [...new Set(graph.nodes.map(node => node.type))];
      document.getElementById("legend").innerHTML = types.map(type =>
        `<span><i class="swatch" style="background:${COLORS[type] || COLORS.other}"></i>${escapeText(type.replaceAll("_", " "))}</span>`
      ).join("");
    }

    function render() {
      const graph = GRAPHS[state.graphIndex];
      const tree = buildTree(graph);
      if (!tree.byId.has(state.zoomId)) state.zoomId = "__root__";
      if (!tree.byId.has(state.selectedId)) state.selectedId = "__root__";
      document.getElementById("taskTitle").textContent = graph.problem_statement || graph.instance_id;
      document.getElementById("taskMeta").textContent = `${graph.instance_id} · ${graph.model}` +
        (graph.resolved == null ? "" : ` · resolved: ${graph.resolved}`);
      document.getElementById("widthHelp").textContent = graph.weight_unit === "estimated_tokens"
        ? "Width represents estimated transcript tokens."
        : "Width represents transcript events.";
      document.getElementById("widthHelp").title = graph.weight_note;
      renderLegend(graph);
      renderChart(tree, graph);
      renderCrumbs(tree);
      renderDetails(tree.byId.get(state.selectedId), graph);
    }
    render();
  </script>
</body>
</html>
'''
