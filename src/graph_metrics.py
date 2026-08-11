#!/usr/bin/env python3
"""
Graph-, vertex-, and edge-level analysis metrics for SSN igraph objects.

Called from plot_ssn.py once per threshold on the graph it has already built —
this module never rebuilds or reloads a graph itself (graph construction is the
expensive step in the pipeline).

The graph passed in is always undirected and simple (self-loops/parallel edges
already removed by plot_ssn.create_graph), so directed-only igraph APIs are not
implemented here: is_dag, feedback_arc_set, topological_sorting, reciprocity,
is_mutual, dyad_census, triad_census. Pairwise similarity/matrix-valued measures
(similarity_dice/jaccard/inverse_log_weighted, cocitation, bibcoupling) are out
of scope (O(V^2) cost, not currently needed).
"""

import csv
import json
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_logger(logger_arg):
    return logger_arg or logger


@contextmanager
def _timed_step(label, log):
    start = time.perf_counter()
    log.info(f"START {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info(f"END   {label}: {elapsed:.2f}s")


PRESETS = {
    "fast": {
        "global": {
            "basic": True, "distributions": False, "connectedness": False,
            "cliques_motifs": False, "separators": False, "optimality": False,
            "independence_sets": False, "other_measures": True, "boolean_props": True,
        },
        "vertex": {"enabled": False, "structural": False},
        "edge": {"enabled": False},
        "components": {"enabled": False},
    },
    "standard": {
        "global": {
            "basic": True, "distributions": True, "connectedness": True,
            "cliques_motifs": False, "separators": False, "optimality": True,
            "independence_sets": False, "other_measures": True, "boolean_props": True,
        },
        "vertex": {"enabled": True, "structural": True},
        "edge": {"enabled": False},
        "components": {"enabled": True},
    },
    "full": {
        "global": {
            "basic": True, "distributions": True, "connectedness": True,
            "cliques_motifs": True, "separators": True, "optimality": True,
            "independence_sets": True, "other_measures": True, "boolean_props": True,
        },
        "vertex": {"enabled": True, "structural": True},
        "edge": {"enabled": True},
        "components": {"enabled": True},
    },
    "custom": {
        "global": {},
        "vertex": {"enabled": False},
        "edge": {"enabled": False},
        "components": {"enabled": False},
    },
}

DEFAULT_GUARDS = {
    "max_nodes_mincut": 5000,
    "max_nodes_cliques": 500,       # per-component cap: cliques/motifs/independence/separators
    "max_cliques_returned": 100000, # cap on maximal_cliques() enumeration size
}


def _deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_metrics_config(user_cfg):
    """Merge a preset (fast|standard|full|custom) with explicit overrides and hard guards."""
    user_cfg = user_cfg or {}
    preset_name = str(user_cfg.get("preset", "standard")).lower()
    preset = PRESETS.get(preset_name, PRESETS["standard"])

    resolved = {
        "enabled": bool(user_cfg.get("enabled", False)),
        "preset": preset_name,
        "guards": {**DEFAULT_GUARDS, **(user_cfg.get("guards") or {})},
        "global": _deep_merge(preset["global"], user_cfg.get("global") or {}),
        "vertex": _deep_merge(preset["vertex"], user_cfg.get("vertex") or {}),
        "edge": _deep_merge(preset["edge"], user_cfg.get("edge") or {}),
        "components": _deep_merge(preset["components"], user_cfg.get("components") or {}),
        "output": {
            "write_vertex_table": True,
            "write_edge_table": False,
            "write_component_table": True,
            "color_with": [],
            **(user_cfg.get("output") or {}),
        },
    }
    resolved["vertex"].setdefault("personalized_pagerank", {"enabled": False, "reset_col": None})
    return resolved


# ── Global metrics ───────────────────────────────────────────────────────────

def compute_global_metrics(g, resolved_cfg, logger=None):
    """Returns {"flat": {...scalars/bools...}, "detail": {...lists/histograms...},
    "skipped": {name: reason}, "errors": {name: str(exc)}}."""
    log = _get_logger(logger)
    flat, detail, skipped, errors = {}, {}, {}, {}
    cfg = resolved_cfg["global"]
    guards = resolved_cfg["guards"]

    if g is None or g.vcount() == 0:
        return {"flat": flat, "detail": detail, "skipped": skipped, "errors": errors}

    def safe(name, fn):
        try:
            return fn()
        except Exception as e:
            log.warning(f"graph_metrics: '{name}' failed: {e}")
            errors[name] = str(e)
            return None

    def skip(name, reason):
        log.warning(f"graph_metrics: skipping '{name}': {reason}")
        skipped[name] = reason

    n = g.vcount()
    names = g.vs["name"]

    if cfg.get("basic"):
        with _timed_step("compute global metrics: basic", log):
            connected = safe("is_connected", g.is_connected)
            target_g = g
            if connected is False:
                components = g.components()
                largest = max(components, key=len)
                target_g = g.induced_subgraph(largest)
                detail["basic_computed_on"] = "largest_component"
                detail["basic_n_components_excluded"] = len(components) - 1
            else:
                detail["basic_computed_on"] = "whole_graph"

            diam = safe("diameter", lambda: target_g.diameter(unconn=True))
            if diam is not None:
                flat["diameter"] = diam
            rad = safe("radius", target_g.radius)
            if rad is not None:
                flat["radius"] = rad
            apl = safe("average_path_length", lambda: target_g.average_path_length(unconn=True))
            if apl is not None:
                flat["average_path_length"] = apl

            girth = safe("girth", g.girth)
            if girth is not None:
                flat["girth"] = None if math.isinf(girth) else girth

    if cfg.get("distributions"):
        with _timed_step("compute global metrics: distributions", log):
            dd = safe("degree_distribution", g.degree_distribution)
            if dd is not None:
                detail["degree_distribution"] = list(dd.bins())
            plh = safe("path_length_hist", g.path_length_hist)
            if plh is not None:
                detail["path_length_hist"] = list(plh.bins())

    if cfg.get("connectedness"):
        with _timed_step("compute global metrics: connectedness", log):
            aps = safe("articulation_points", g.articulation_points)
            if aps is not None:
                flat["n_articulation_points"] = len(aps)
                detail["articulation_points"] = [names[i] for i in aps]
            bc = safe("biconnected_components", g.biconnected_components)
            if bc is not None:
                flat["n_biconnected_components"] = len(bc)
                detail["biconnected_components"] = [[names[i] for i in comp] for comp in bc]

    if cfg.get("separators"):
        with _timed_step("compute global metrics: separators", log):
            max_nodes = guards["max_nodes_cliques"]
            if n > max_nodes:
                skip("separators", f"graph has {n} nodes > guard max_nodes_cliques={max_nodes}")
            else:
                seps = safe("all_minimal_st_separators", g.all_minimal_st_separators)
                if seps is not None:
                    flat["n_minimal_st_separators"] = len(seps)
                    detail["minimal_st_separators"] = [[names[i] for i in s] for s in seps]
                mss = safe("minimum_size_separators", g.minimum_size_separators)
                if mss is not None:
                    flat["n_minimum_size_separators"] = len(mss)
                    detail["minimum_size_separators"] = [[names[i] for i in s] for s in mss]

    if cfg.get("cliques_motifs"):
        with _timed_step("compute global metrics: cliques_motifs", log):
            max_nodes = guards["max_nodes_cliques"]
            max_results = guards["max_cliques_returned"]
            components = g.components()
            clique_numbers = []
            maximal_cliques_named = []
            largest_cliques_named = []
            for comp in components:
                if len(comp) > max_nodes:
                    skip(f"cliques_motifs(component size {len(comp)})",
                         f"component exceeds guard max_nodes_cliques={max_nodes}")
                    continue
                sub = g.induced_subgraph(comp)
                cn = safe(f"clique_number(component size {len(comp)})", sub.clique_number)
                if cn is not None:
                    clique_numbers.append(cn)
                mc = safe(f"maximal_cliques(component size {len(comp)})",
                          lambda sub=sub: sub.maximal_cliques(max_results=max_results))
                if mc is not None:
                    maximal_cliques_named.extend([comp[i] for i in clique] for clique in mc)
                lc = safe(f"largest_cliques(component size {len(comp)})", sub.largest_cliques)
                if lc is not None:
                    largest_cliques_named.extend([comp[i] for i in clique] for clique in lc)
            if clique_numbers:
                flat["clique_number"] = max(clique_numbers)
            if maximal_cliques_named:
                flat["n_maximal_cliques"] = len(maximal_cliques_named)
                detail["maximal_cliques"] = [[names[i] for i in clique] for clique in maximal_cliques_named]
            if largest_cliques_named:
                detail["largest_cliques"] = [[names[i] for i in clique] for clique in largest_cliques_named]

            motif_size = 3
            motifs = safe("motifs_randesu", lambda: g.motifs_randesu(size=motif_size))
            if motifs is not None:
                detail["motifs_randesu"] = [
                    None if (isinstance(v, float) and math.isnan(v)) else v for v in motifs
                ]

    if cfg.get("optimality"):
        with _timed_step("compute global metrics: optimality", log):
            fp = safe("farthest_points", g.farthest_points)
            if fp is not None and fp[0] is not None:
                flat["farthest_points_distance"] = fp[2]
                detail["farthest_points"] = [names[fp[0]], names[fp[1]]]

            max_nodes_mincut = guards["max_nodes_mincut"]
            if n > max_nodes_mincut:
                skip("mincut", f"graph has {n} nodes > guard max_nodes_mincut={max_nodes_mincut}")
            else:
                mc = safe("mincut", g.mincut)
                if mc is not None:
                    flat["mincut_value"] = mc.value

            # g.modularity() needs a membership vector; Louvain (community_multilevel)
            # is used here to derive one, purely to report modularity/community count.
            vc = safe("community_multilevel", g.community_multilevel)
            if vc is not None:
                flat["modularity"] = vc.modularity
                flat["n_communities_detected"] = len(set(vc.membership))
                # Consumed by compute_graph_metrics() to also expose community_id
                # as a per-vertex value; not written to the detail JSON as-is.
                detail["_community_membership"] = vc.membership

    if cfg.get("independence_sets"):
        with _timed_step("compute global metrics: independence_sets", log):
            max_nodes = guards["max_nodes_cliques"]
            components = g.components()
            alphas = []
            maximal_ivs_named = []
            largest_ivs_named = []
            for comp in components:
                if len(comp) > max_nodes:
                    skip(f"independence_sets(component size {len(comp)})",
                         f"component exceeds guard max_nodes_cliques={max_nodes}")
                    continue
                sub = g.induced_subgraph(comp)
                alpha = safe(f"independence_number(component size {len(comp)})", sub.independence_number)
                if alpha is not None:
                    alphas.append(alpha)
                miv = safe(f"maximal_independent_vertex_sets(component size {len(comp)})",
                           sub.maximal_independent_vertex_sets)
                if miv is not None:
                    maximal_ivs_named.extend([comp[i] for i in s] for s in miv)
                liv = safe(f"largest_independent_vertex_sets(component size {len(comp)})",
                           sub.largest_independent_vertex_sets)
                if liv is not None:
                    largest_ivs_named.extend([comp[i] for i in s] for s in liv)
            if alphas:
                flat["independence_number"] = sum(alphas)
            if maximal_ivs_named:
                flat["n_maximal_independent_vertex_sets"] = len(maximal_ivs_named)
                detail["maximal_independent_vertex_sets"] = [
                    [names[i] for i in s] for s in maximal_ivs_named
                ]
            if largest_ivs_named:
                detail["largest_independent_vertex_sets"] = [
                    [names[i] for i in s] for s in largest_ivs_named
                ]

    if cfg.get("other_measures"):
        with _timed_step("compute global metrics: other_measures", log):
            ad = safe("assortativity_degree", g.assortativity_degree)
            if ad is not None:
                flat["assortativity_degree"] = ad
            dens = safe("density", g.density)
            if dens is not None:
                flat["density"] = dens
            tu = safe("transitivity_undirected", lambda: g.transitivity_undirected(mode="zero"))
            if tu is not None:
                flat["transitivity_undirected"] = tu
            tal = safe("transitivity_avglocal_undirected",
                       lambda: g.transitivity_avglocal_undirected(mode="zero"))
            if tal is not None:
                flat["transitivity_avglocal_undirected"] = tal

    if cfg.get("boolean_props"):
        with _timed_step("compute global metrics: boolean_props", log):
            b = safe("is_bipartite", g.is_bipartite)
            if b is not None:
                flat["is_bipartite"] = b
            b = safe("is_connected", g.is_connected)
            if b is not None:
                flat["is_connected"] = b
            b = safe("is_simple", g.is_simple)
            if b is not None:
                flat["is_simple"] = b
            b = safe("has_multiple", g.has_multiple)
            if b is not None:
                flat["has_multiple"] = b
            # NOTE: igraph's is_weighted() only checks for an edge attribute literally
            # named "weight" — this codebase stores similarity as "pident" (see
            # plot_ssn.create_graph), so is_weighted() always reports False here.
            # has_pident_weights is the correct signal for this graph; both are kept
            # so this discrepancy isn't "fixed" incorrectly later.
            b = safe("is_weighted", g.is_weighted)
            if b is not None:
                flat["is_weighted"] = b
            flat["has_pident_weights"] = "pident" in g.es.attributes() if g.ecount() else False
            flat["is_directed"] = False  # always False: create_graph() always builds directed=False

    return {"flat": flat, "detail": detail, "skipped": skipped, "errors": errors}


# ── Vertex metrics ───────────────────────────────────────────────────────────

def compute_vertex_metrics(g, resolved_cfg, logger=None, metadata=None, id_col=None):
    """Returns {"vector": {metric_name: {vertex_name: value}}, "skipped": {...}, "errors": {...}}."""
    log = _get_logger(logger)
    vector, skipped, errors = {}, {}, {}
    cfg = resolved_cfg["vertex"]

    if g is None or g.vcount() == 0 or not cfg.get("enabled"):
        return {"vector": vector, "skipped": skipped, "errors": errors}

    names = g.vs["name"]

    def safe(name, fn):
        try:
            return dict(zip(names, fn()))
        except Exception as e:
            log.warning(f"graph_metrics: '{name}' failed: {e}")
            errors[name] = str(e)
            return None

    if cfg.get("structural"):
        with _timed_step("compute vertex metrics: structural", log):
            has_pident = "pident" in g.es.attributes()
            metrics = [
                ("authority_score", g.authority_score),
                ("hub_score", g.hub_score),
                ("betweenness", g.betweenness),
                ("closeness", g.closeness),
                ("constraint", g.constraint),
                ("coreness", g.coreness),
                ("eccentricity", g.eccentricity),
                ("eigenvector_centrality", g.eigenvector_centrality),
                ("harmonic_centrality", g.harmonic_centrality),
                ("pagerank", g.pagerank),
                ("strength", lambda: g.strength(weights="pident" if has_pident else None)),
                ("transitivity_local_undirected", lambda: g.transitivity_local_undirected(mode="zero")),
            ]
            for metric_name, fn in metrics:
                result = safe(metric_name, fn)
                if result is not None:
                    vector[metric_name] = result

    pp_cfg = cfg.get("personalized_pagerank") or {}
    if pp_cfg.get("enabled"):
        with _timed_step("compute vertex metrics: personalized_pagerank", log):
            reset_ids = pp_cfg.get("reset_ids")
            reset_col = pp_cfg.get("reset_col")
            if not reset_ids and reset_col and metadata is not None and id_col is not None \
                    and reset_col in metadata.columns:
                seed_rows = metadata[metadata[reset_col].astype(bool)]
                reset_ids = seed_rows[id_col].astype(str).tolist()

            name_to_idx = {name: i for i, name in enumerate(names)}
            reset_vertices = [name_to_idx[i] for i in (reset_ids or []) if i in name_to_idx]
            if reset_vertices:
                result = safe("personalized_pagerank",
                              lambda: g.personalized_pagerank(reset_vertices=reset_vertices))
                if result is not None:
                    vector["personalized_pagerank"] = result
            else:
                reason = "no reset_ids/reset_col resolved to any vertex in the graph"
                log.warning(f"graph_metrics: skipping 'personalized_pagerank': {reason}")
                skipped["personalized_pagerank"] = reason

    return {"vector": vector, "skipped": skipped, "errors": errors}


# ── Edge metrics ──────────────────────────────────────────────────────────────

def compute_edge_metrics(g, resolved_cfg, logger=None):
    """Returns {"vector": {metric_name: [values aligned to g.es order]}, "skipped": {...}, "errors": {...}}."""
    log = _get_logger(logger)
    vector, skipped, errors = {}, {}, {}
    cfg = resolved_cfg["edge"]

    if g is None or g.ecount() == 0 or not cfg.get("enabled"):
        return {"vector": vector, "skipped": skipped, "errors": errors}

    def safe(name, fn):
        try:
            return list(fn())
        except Exception as e:
            log.warning(f"graph_metrics: '{name}' failed: {e}")
            errors[name] = str(e)
            return None

    with _timed_step("compute edge metrics", log):
        for metric_name, fn in [
            ("is_loop", g.is_loop),
            ("is_multiple", g.is_multiple),
            ("count_multiple", g.count_multiple),
            ("edge_betweenness", g.edge_betweenness),
        ]:
            result = safe(metric_name, fn)
            if result is not None:
                vector[metric_name] = result

    return {"vector": vector, "skipped": skipped, "errors": errors}


# ── Component metrics ─────────────────────────────────────────────────────────

def compute_component_metrics(g, resolved_cfg, logger=None):
    """Returns {"table": [ {component_id, n_nodes, n_edges, ...}, ... ],
    "skipped": {metric_name: {"count": n, "reason": str}},
    "errors": {metric_name: {"count": n, "example": str}}} — one row per connected
    component, reusing the same `global` category toggles/guards as compute_global_metrics."""
    log = _get_logger(logger)
    table = []
    skipped_counts, errors_counts = {}, {}
    cfg = resolved_cfg["components"]
    gcfg = resolved_cfg["global"]
    guards = resolved_cfg["guards"]

    if g is None or g.vcount() == 0 or not cfg.get("enabled"):
        return {"table": table, "skipped": skipped_counts, "errors": errors_counts}

    def record_skip(name, reason):
        entry = skipped_counts.setdefault(name, {"count": 0, "reason": reason})
        entry["count"] += 1

    def safe(row, name, fn):
        try:
            row[name] = fn()
        except Exception as e:
            log.warning(f"graph_metrics: component metric '{name}' failed: {e}")
            entry = errors_counts.setdefault(name, {"count": 0, "example": str(e)})
            entry["count"] += 1

    max_nodes_cliques = guards["max_nodes_cliques"]
    max_nodes_mincut = guards["max_nodes_mincut"]
    max_cliques_returned = guards["max_cliques_returned"]

    with _timed_step(f"compute component metrics ({g.vcount()} vertices)", log):
        for comp_id, comp in enumerate(g.components()):
            sub = g.induced_subgraph(comp)
            row = {"component_id": comp_id, "n_nodes": sub.vcount(), "n_edges": sub.ecount()}

            if gcfg.get("basic"):
                safe(row, "diameter", lambda: sub.diameter(unconn=True))
                safe(row, "radius", sub.radius)
                safe(row, "average_path_length", lambda: sub.average_path_length(unconn=True))
                def _girth():
                    v = sub.girth()
                    return None if math.isinf(v) else v
                safe(row, "girth", _girth)

            if gcfg.get("connectedness"):
                safe(row, "n_articulation_points", lambda: len(sub.articulation_points()))
                safe(row, "n_biconnected_components", lambda: len(sub.biconnected_components()))

            if gcfg.get("separators"):
                if sub.vcount() > max_nodes_cliques:
                    reason = f"component exceeds guard max_nodes_cliques={max_nodes_cliques}"
                    record_skip("n_minimal_st_separators", reason)
                    record_skip("n_minimum_size_separators", reason)
                else:
                    safe(row, "n_minimal_st_separators", lambda: len(sub.all_minimal_st_separators()))
                    safe(row, "n_minimum_size_separators", lambda: len(sub.minimum_size_separators()))

            if gcfg.get("cliques_motifs"):
                if sub.vcount() > max_nodes_cliques:
                    reason = f"component exceeds guard max_nodes_cliques={max_nodes_cliques}"
                    record_skip("clique_number", reason)
                    record_skip("n_maximal_cliques", reason)
                else:
                    safe(row, "clique_number", sub.clique_number)
                    safe(row, "n_maximal_cliques",
                         lambda: len(sub.maximal_cliques(max_results=max_cliques_returned)))

            if gcfg.get("independence_sets"):
                if sub.vcount() > max_nodes_cliques:
                    reason = f"component exceeds guard max_nodes_cliques={max_nodes_cliques}"
                    record_skip("independence_number", reason)
                    record_skip("n_maximal_independent_vertex_sets", reason)
                else:
                    safe(row, "independence_number", sub.independence_number)
                    safe(row, "n_maximal_independent_vertex_sets",
                         lambda: len(sub.maximal_independent_vertex_sets()))

            if gcfg.get("optimality"):
                if sub.vcount() > max_nodes_mincut:
                    record_skip("mincut_value", f"component exceeds guard max_nodes_mincut={max_nodes_mincut}")
                else:
                    safe(row, "mincut_value", lambda: sub.mincut().value)

                def _modularity():
                    vc = sub.community_multilevel()
                    return vc.modularity, len(set(vc.membership))
                try:
                    row["modularity"], row["n_communities_detected"] = _modularity()
                except Exception as e:
                    log.warning(f"graph_metrics: component metric 'community_multilevel' failed: {e}")
                    entry = errors_counts.setdefault("community_multilevel", {"count": 0, "example": str(e)})
                    entry["count"] += 1

            if gcfg.get("other_measures"):
                safe(row, "assortativity_degree", sub.assortativity_degree)
                safe(row, "density", sub.density)
                safe(row, "transitivity_undirected", lambda: sub.transitivity_undirected(mode="zero"))
                safe(row, "transitivity_avglocal_undirected",
                     lambda: sub.transitivity_avglocal_undirected(mode="zero"))

            if gcfg.get("boolean_props"):
                safe(row, "is_bipartite", sub.is_bipartite)

            table.append(row)

    return {"table": table, "skipped": skipped_counts, "errors": errors_counts}


# ── Output writer ─────────────────────────────────────────────────────────────

def write_outputs(g, output_base, thresh, global_result, vertex_result, edge_result,
                   component_result, resolved_cfg, logger=None):
    """Writes whichever files are enabled; returns {"files": {...paths...}}."""
    log = _get_logger(logger)
    output_cfg = resolved_cfg.get("output", {})
    files = {}

    detail_payload = {
        "detail": global_result["detail"],
        "skipped": {**global_result["skipped"], **vertex_result["skipped"], **edge_result["skipped"]},
        "errors": {**global_result["errors"], **vertex_result["errors"], **edge_result["errors"]},
        # Component-level skips/errors are aggregated by metric name across all
        # components (not one entry per component_id — see compute_component_metrics),
        # so they're kept as separate, differently-shaped blocks here.
        "component_skipped": component_result.get("skipped", {}),
        "component_errors": component_result.get("errors", {}),
    }
    metrics_file = f"{output_base}_t{thresh}_graph_metrics.json"
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(detail_payload, f, indent=2, default=str)
    files["graph_metrics_json"] = metrics_file
    log.info(f"Saved: {metrics_file}")

    if vertex_result["vector"] and output_cfg.get("write_vertex_table", True):
        vertex_file = f"{output_base}_t{thresh}_vertex_metrics.tsv"
        names = g.vs["name"]
        metric_names = sorted(vertex_result["vector"].keys())
        with open(vertex_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["name"] + metric_names)
            for name in names:
                row = [name] + [vertex_result["vector"][m].get(name, "") for m in metric_names]
                writer.writerow(row)
        files["vertex_metrics_tsv"] = vertex_file
        log.info(f"Saved: {vertex_file}")

    if edge_result["vector"] and output_cfg.get("write_edge_table", False):
        edge_file = f"{output_base}_t{thresh}_edge_metrics.tsv"
        names = g.vs["name"]
        metric_names = sorted(edge_result["vector"].keys())
        pident_attr = g.es["pident"] if "pident" in g.es.attributes() else [None] * g.ecount()
        with open(edge_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source", "target", "pident"] + metric_names)
            for i, edge in enumerate(g.es):
                row = [names[edge.source], names[edge.target], pident_attr[i]] + [
                    edge_result["vector"][m][i] for m in metric_names
                ]
                writer.writerow(row)
        files["edge_metrics_tsv"] = edge_file
        log.info(f"Saved: {edge_file}")

    if component_result["table"] and output_cfg.get("write_component_table", True):
        component_file = f"{output_base}_t{thresh}_component_metrics.tsv"
        rows = component_result["table"]
        fixed_cols = ["component_id", "n_nodes", "n_edges"]
        other_cols = sorted({key for row in rows for key in row if key not in fixed_cols})
        columns = fixed_cols + other_cols
        with open(component_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row.get(c, "") for c in columns])
        files["component_metrics_tsv"] = component_file
        log.info(f"Saved: {component_file}")

    return files


# ── Orchestrator ──────────────────────────────────────────────────────────────

def compute_graph_metrics(g, thresh, output_base, metrics_cfg, *, metadata=None,
                           id_col=None, logger=None):
    """
    Single entry point plot_ssn.py calls, once per threshold, right after graph_stats().

    Returns {"global_flat": {...}, "vertex_metrics": {metric: {name: value}},
             "files": {...}, "skipped": {...}, "errors": {...}}.
    """
    log = _get_logger(logger)
    resolved_cfg = resolve_metrics_config(metrics_cfg)

    if g is None or g.vcount() == 0 or not resolved_cfg["enabled"]:
        return {"global_flat": {}, "vertex_metrics": {}, "files": {}, "skipped": {}, "errors": {}}

    global_result = compute_global_metrics(g, resolved_cfg, logger=log)
    vertex_result = compute_vertex_metrics(g, resolved_cfg, logger=log, metadata=metadata, id_col=id_col)
    edge_result = compute_edge_metrics(g, resolved_cfg, logger=log)
    component_result = compute_component_metrics(g, resolved_cfg, logger=log)

    # Community membership is derived as part of the global "optimality" category
    # (needed there to compute modularity); surface it as a per-vertex value too
    # so it can double as a color-by option.
    membership = global_result["detail"].pop("_community_membership", None)
    if membership is not None:
        names = g.vs["name"]
        vertex_result["vector"]["community_id"] = dict(zip(names, membership))

    # Connected-component membership, so vertex_metrics.tsv rows can be joined
    # against component_metrics.tsv rows without re-deriving membership.
    if resolved_cfg["components"].get("enabled"):
        names = g.vs["name"]
        vertex_result["vector"]["component_id"] = dict(zip(names, g.components().membership))

    files = write_outputs(g, output_base, thresh, global_result, vertex_result, edge_result,
                           component_result, resolved_cfg, logger=log)

    skipped = {**global_result["skipped"], **vertex_result["skipped"], **edge_result["skipped"]}
    errors = {**global_result["errors"], **vertex_result["errors"], **edge_result["errors"]}

    return {
        "global_flat": global_result["flat"],
        "vertex_metrics": vertex_result["vector"],
        "files": files,
        "skipped": skipped,
        "errors": errors,
    }
