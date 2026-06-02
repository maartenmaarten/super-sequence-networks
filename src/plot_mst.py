#!/usr/bin/env python3
"""
Minimum Spanning Tree (MST) SSN plotter using Plotly.

Reads a YAML config, filters BLAST results by threshold, computes a minimum
spanning tree per connected component via scipy.sparse.csgraph.minimum_spanning_tree,
and generates interactive HTML plots.

Usage:
    python plot_mst.py [config.yaml]
"""

import math
import sys
import argparse
from pathlib import Path
import json
import time
import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import yaml
import igraph as ig
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

try:
    import kaleido
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False

logger = logging.getLogger(__name__)


@contextmanager
def timed_step(label):
    start = time.perf_counter()
    logger.info(f"START {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"END   {label}: {elapsed:.2f}s")


def normalize_ids(ids):
    normalized = []
    for idx in ids:
        idx = str(idx)
        if "|" in idx:
            parts = idx.split("|")
            if len(parts) >= 2:
                idx = parts[1]
        if idx.isdigit():
            idx = f"{idx}.0"
        normalized.append(idx)
    return normalized


def build_color_lookup(meta, id_col, color_col, clusters=None, pazy_ids=None, id_map=None):
    if meta is None or id_col is None or color_col not in meta.columns:
        logger.warning(f"Missing metadata or color column '{color_col}' not found. All nodes will be 'Unknown'.")
        return {}

    lookup = dict(zip(meta[id_col].astype(str), meta[color_col].astype(str)))

    if id_map is not None and len(id_map) > 0:
        for _, row in id_map.iterrows():
            if str(row["genbank"]) in lookup:
                lookup[str(row["uniprot"])] = lookup[str(row["genbank"])]
        logger.info(f"Extended lookup with {len(id_map)} UniProt keys")

    if clusters is not None and len(clusters) > 0:
        cluster_annots = {}
        for _, row in clusters.iterrows():
            rep = str(row["rep"])
            member = str(row["member"])
            member_normalized = normalize_ids([member])[0]

            annot = None
            if member_normalized in lookup:
                annot = lookup[member_normalized]
            elif member in lookup:
                annot = lookup[member]

            if annot and annot != "":
                if rep not in cluster_annots:
                    cluster_annots[rep] = {}
                cluster_annots[rep][annot] = cluster_annots[rep].get(annot, 0) + 1

        for rep, annot_counts in cluster_annots.items():
            majority = max(annot_counts, key=annot_counts.get)
            lookup[rep] = majority

        logger.info(f"Propagated cluster annotations for {len(cluster_annots)} representatives")

    return lookup


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def infer_config_base(config_path):
    config_dir = config_path.resolve().parent
    if config_dir.name == "logs":
        return config_dir.parent
    return config_dir


def _yn(value):
    """Treat YAML string 'None'/'null'/'~'/'' as Python None."""
    if value is None:
        return None
    if str(value).strip().lower() in ("none", "null", "~", ""):
        return None
    return value


def resolve_path(path_value, config_base):
    path_value = _yn(path_value)
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    resolved = config_base / path
    if resolved.exists():
        return resolved
    return resolved


def _is_continuous(values):
    """Return True when all non-sentinel values can be parsed as floats."""
    sentinels = {"Unknown", "All sequences", None, ""}
    candidates = [v for v in values if v not in sentinels]
    if not candidates:
        return False
    try:
        [float(v) for v in candidates]
        return True
    except (ValueError, TypeError):
        return False


def load_data(tsv_file):
    with timed_step(f"load similarity table {tsv_file}"):
        df = pd.read_csv(
            tsv_file,
            sep="\t",
            names=["qseqid", "sseqid", "pident", "length", "mismatch",
                   "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"],
            dtype={"qseqid": str, "sseqid": str, "pident": float}
        )
    logger.info(f"Loaded {len(df)} rows from {tsv_file}")

    with timed_step("remove self-hits"):
        self_hit_mask = df["qseqid"] == df["sseqid"]
        df_no_self = df[~self_hit_mask].copy()
    logger.info(f"Removed self-hits: {int(self_hit_mask.sum())} rows")
    logger.info(f"After removing self-hits: {len(df_no_self)} rows")
    print(f"Loaded data: {len(df_no_self)} edges")

    return df_no_self


def build_mst_graph(df_thresh, all_nodes=None):
    """
    Build a minimum spanning tree (forest) from threshold-filtered edges.

    Uses distance = 100 - pident as edge weight so that MST retains
    the highest-similarity edges. Handles disconnected graphs by computing
    a minimum spanning forest.
    """
    if len(df_thresh) == 0:
        if all_nodes is not None and len(all_nodes) > 0:
            with timed_step("build igraph with isolated nodes (no edges)"):
                g = ig.Graph(len(all_nodes), directed=False)
                g.vs["name"] = all_nodes
            return g
        return None

    with timed_step("deduplicate undirected edges"):
        df_edges = df_thresh[["qseqid", "sseqid", "pident"]].copy()
        df_edges["source"] = df_edges[["qseqid", "sseqid"]].min(axis=1)
        df_edges["target"] = df_edges[["qseqid", "sseqid"]].max(axis=1)
        before_dedup = len(df_edges)
        df_edges = (
            df_edges
            .groupby(["source", "target"], as_index=False)["pident"]
            .max()
        )
        logger.info(f"Deduplicated edges: {before_dedup} -> {len(df_edges)}")

    with timed_step("build node index"):
        # Collect all nodes that appear in edges, plus any extra all_nodes
        edge_nodes = sorted(set(df_edges["source"]) | set(df_edges["target"]))
        if all_nodes is not None:
            combined = sorted(set(edge_nodes) | set(all_nodes))
        else:
            combined = edge_nodes
        node_index = {name: i for i, name in enumerate(combined)}
        n = len(combined)

    with timed_step("build sparse distance matrix"):
        rows = df_edges["source"].map(node_index).values
        cols = df_edges["target"].map(node_index).values
        # Distance = 100 - pident  (lower = more similar = preferred by MST)
        # Add small epsilon so zero-distance edges (100% identity) are still included
        distances = (100.0 - df_edges["pident"].values) + 1e-6

        # Build symmetric CSR matrix (only upper triangle needed by minimum_spanning_tree)
        sparse = csr_matrix(
            (distances, (rows, cols)),
            shape=(n, n)
        )

    with timed_step("compute minimum spanning tree (forest)"):
        mst_sparse = minimum_spanning_tree(sparse)

    with timed_step("extract MST edges"):
        mst_coo = mst_sparse.tocoo()
        mst_rows = mst_coo.row
        mst_cols = mst_coo.col
        mst_distances = mst_coo.data
        mst_pident = 100.0 - mst_distances + 1e-6  # recover approximate pident

        node_names = combined
        mst_edges = [
            (node_names[r], node_names[c], float(p))
            for r, c, p in zip(mst_rows, mst_cols, mst_pident)
        ]
        logger.info(f"MST has {len(mst_edges)} edges (from {len(df_edges)} input edges)")

    with timed_step("build igraph from MST edges"):
        if len(mst_edges) == 0:
            g = ig.Graph(n, directed=False)
            g.vs["name"] = node_names
            return g

        sources = [e[0] for e in mst_edges]
        targets = [e[1] for e in mst_edges]
        pidents = [e[2] for e in mst_edges]

        edge_tuples = list(zip(sources, targets))
        g = ig.Graph.TupleList(edge_tuples, directed=False)
        g.es["pident"] = pidents

    # Add isolated nodes not reached by any MST edge
    if all_nodes is not None:
        existing = set(g.vs["name"])
        isolated = [node for node in all_nodes if node not in existing]
        if isolated:
            with timed_step(f"add {len(isolated)} isolated nodes"):
                start_idx = g.vcount()
                g.add_vertices(len(isolated))
                for i, name in enumerate(isolated):
                    g.vs[start_idx + i]["name"] = name
            logger.info(f"Added {len(isolated)} isolated nodes")
    elif len(node_names) > g.vcount():
        # Nodes in the index but not in MST edges (fully isolated within component)
        existing = set(g.vs["name"])
        isolated = [n for n in node_names if n not in existing]
        if isolated:
            with timed_step(f"add {len(isolated)} isolated nodes from edge list"):
                start_idx = g.vcount()
                g.add_vertices(len(isolated))
                for i, name in enumerate(isolated):
                    g.vs[start_idx + i]["name"] = name

    return g


def exclude_singleton_components(g):
    if g is None or g.vcount() == 0:
        return g, 0

    components = g.components()
    singleton_vertices = [component[0] for component in components if len(component) == 1]
    if not singleton_vertices:
        return g, 0

    singleton_set = set(singleton_vertices)
    keep_vertices = [idx for idx in range(g.vcount()) if idx not in singleton_set]
    if not keep_vertices:
        return None, len(singleton_vertices)

    return g.induced_subgraph(keep_vertices), len(singleton_vertices)


def get_component_grid_layout(g):
    components = sorted(g.components(), key=len, reverse=True)
    coords = [None] * g.vcount()

    component_specs = []
    total_area = 0.0
    for component in components:
        n_nodes = len(component)
        radius = max(1.0, n_nodes ** 0.5)
        diameter = radius * 2
        component_specs.append((component, radius, diameter))
        total_area += diameter * diameter

    padding = 3.0
    target_row_width = max(10.0, total_area ** 0.5 * 1.4)
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0

    for component, radius, diameter in component_specs:
        if x_cursor > 0 and x_cursor + diameter > target_row_width:
            x_cursor = 0.0
            y_cursor += row_height + padding
            row_height = 0.0

        subgraph = g.induced_subgraph(component)
        if subgraph.vcount() == 1:
            local_coords = [(0.0, 0.0)]
        elif subgraph.vcount() <= 30:
            local_layout = subgraph.layout("circle")
            local_coords = [(point[0], point[1]) for point in local_layout]
        else:
            try:
                local_layout = subgraph.layout("lgl")
            except Exception:
                local_layout = subgraph.layout("fr")
            local_coords = [(point[0], point[1]) for point in local_layout]

        local_x = [point[0] for point in local_coords]
        local_y = [point[1] for point in local_coords]
        center_x = (min(local_x) + max(local_x)) / 2
        center_y = (min(local_y) + max(local_y)) / 2
        local_width = max(max(local_x) - min(local_x), max(local_y) - min(local_y), 1.0)
        scale = diameter / local_width

        offset_x = x_cursor + radius
        offset_y = -(y_cursor + radius)
        for local_idx, vertex_idx in enumerate(component):
            x, y = local_coords[local_idx]
            coords[vertex_idx] = (
                (x - center_x) * scale + offset_x,
                (y - center_y) * scale + offset_y,
            )

        x_cursor += diameter + padding
        row_height = max(row_height, diameter)

    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return xs, ys


def get_layout(g, layout_method="component_grid"):
    layout_aliases = {
        "fast": "component_grid",
        "components": "component_grid",
        "component-grid": "component_grid",
        "large": "lgl",
        "large_graph": "lgl",
        "fruchterman_reingold": "fr",
        "fruchterman-reingold": "fr",
        "kamada_kawai": "kk",
        "kamada-kawai": "kk",
    }
    method = layout_aliases.get(str(layout_method).lower(), str(layout_method).lower())

    with timed_step(f"compute {method} layout ({g.vcount()} nodes, {g.ecount()} edges)"):
        if method == "component_grid":
            xs, ys = get_component_grid_layout(g)
        else:
            try:
                layout = g.layout(method)
            except Exception as e:
                logger.warning(f"layout '{layout_method}' failed ({e}); falling back to 'component_grid'")
                return get_component_grid_layout(g)
            xs = [coord[0] for coord in layout]
            ys = [coord[1] for coord in layout]
    return xs, ys


def plot_mst(g, output_file, color_col, thresh, tsv_file,
             color_lookup=None, node_size=5, layout_coords=None, metadata=None, id_col=None):
    if g is None or g.vcount() == 0:
        logger.info("Skipping plot (no nodes)")
        return

    if layout_coords is None:
        xs, ys = get_layout(g)
    else:
        xs, ys = layout_coords

    with timed_step(f"prepare node colors for {color_col}"):
        node_names = g.vs["name"]
        if color_col is None:
            node_colors = ["All sequences"] * len(node_names)
        else:
            node_colors = []
            for name in node_names:
                if color_lookup and name in color_lookup:
                    node_colors.append(color_lookup[name])
                else:
                    node_colors.append("Unknown")

    with timed_step(f"build edge trace for {color_col}"):
        edge_x = []
        edge_y = []
        edge_hovertext = []

        for edge in g.es:
            x0, y0 = xs[edge.source], ys[edge.source]
            x1, y1 = xs[edge.target], ys[edge.target]
            pident = edge["pident"] if "pident" in edge.attributes() else 0

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_hovertext.extend([
                f"{node_names[edge.source]} → {node_names[edge.target]}: {pident:.1f}%",
                f"{node_names[edge.source]} → {node_names[edge.target]}: {pident:.1f}%",
                None
            ])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="rgba(100,100,100,0.4)"),
            hoverinfo="text",
            hovertext=edge_hovertext,
            showlegend=False
        )

    marker_size = max(float(node_size), 3.0) * 2
    continuous = _is_continuous(node_colors)

    def _hover(i):
        node_name = node_names[i]
        lines = [f"<b>{node_name}</b>"]
        if metadata is not None and id_col is not None:
            meta_row = metadata[metadata[id_col].astype(str) == str(node_name)]
            if not meta_row.empty:
                for col in metadata.columns:
                    if col != id_col:
                        lines.append(f"{col}: {meta_row[col].iloc[0]}")
            else:
                lines.append("No annotation found")
        else:
            lines.append(f"{color_col}: {node_colors[i]}")
        return "<br>".join(lines)

    with timed_step(f"build node traces for {color_col} ({'continuous' if continuous else 'discrete'})"):
        node_traces = []

        if continuous:
            float_colors = []
            for c in node_colors:
                try:
                    float_colors.append(float(c))
                except (ValueError, TypeError):
                    float_colors.append(float("nan"))

            known_idx   = [i for i, v in enumerate(float_colors) if not math.isnan(v)]
            unknown_idx = [i for i, v in enumerate(float_colors) if math.isnan(v)]

            if known_idx:
                node_traces.append(go.Scatter(
                    x=[xs[i] for i in known_idx],
                    y=[ys[i] for i in known_idx],
                    mode="markers",
                    name=str(color_col),
                    marker=dict(
                        size=marker_size,
                        color=[float_colors[i] for i in known_idx],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=str(color_col), thickness=15),
                        line=dict(width=0.4, color="white"),
                        opacity=0.9,
                    ),
                    text=[_hover(i) for i in known_idx],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ))
            if unknown_idx:
                node_traces.append(go.Scatter(
                    x=[xs[i] for i in unknown_idx],
                    y=[ys[i] for i in unknown_idx],
                    mode="markers",
                    name="Unknown",
                    marker=dict(
                        size=marker_size,
                        color="rgb(128, 128, 128)",
                        line=dict(width=0.4, color="white"),
                        opacity=0.9,
                    ),
                    text=[_hover(i) for i in unknown_idx],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                ))

        else:
            color_scheme = (
                px.colors.qualitative.Bold
                + px.colors.qualitative.Light24
                + px.colors.qualitative.Dark24
            )
            sentinels = {"Unknown", "All sequences"}
            unique_colors = sorted(c for c in set(node_colors) if c not in sentinels)
            for special in ("All sequences", "Unknown"):
                if special in node_colors:
                    unique_colors.append(special)

            color_map = {
                c: color_scheme[i % len(color_scheme)]
                for i, c in enumerate(unique_colors)
            }
            color_map["Unknown"] = "rgb(128, 128, 128)"
            color_map["All sequences"] = "rgb(31, 119, 180)"

            for color_value in unique_colors:
                indices = [i for i, v in enumerate(node_colors) if v == color_value]
                node_traces.append(go.Scatter(
                    x=[xs[i] for i in indices],
                    y=[ys[i] for i in indices],
                    mode="markers",
                    name=str(color_value),
                    marker=dict(
                        size=marker_size,
                        color=color_map.get(color_value, "grey"),
                        line=dict(width=0.4, color="white"),
                        opacity=0.9,
                    ),
                    text=[_hover(i) for i in indices],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                ))

    with timed_step(f"assemble figure for {color_col}"):
        fig = go.Figure(data=[edge_trace] + node_traces)
        fig.update_layout(
            title={
                "text": (
                    f"{Path(tsv_file).name} - Nodes: {g.vcount()}<br>"
                    f"Minimum Spanning Tree (edges pident >= {thresh})"
                ),
                "x": 0.5,
                "xanchor": "center"
            },
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1200,
            height=1000
        )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with timed_step(f"write HTML {output_file}"):
        fig.write_html(output_file)
    logger.info(f"Saved: {output_file}")
    print(f"  Saved: {output_file}")

    if HAS_KALEIDO:
        png_file = str(output_file).replace(".html", ".png")
        try:
            with timed_step(f"write PNG {png_file}"):
                fig.write_image(png_file, width=1200, height=1000, scale=2)
            logger.info(f"Saved: {png_file}")
            print(f"  Saved: {png_file}")
        except Exception as e:
            logger.warning(f"PNG export failed ({e})")


def count_annotated_nodes(g, metadata, id_col):
    if g is None or metadata is None or id_col is None:
        return 0
    node_names = set(g.vs["name"])
    meta_ids = set(metadata[id_col].astype(str))
    return len(node_names & meta_ids)


def graph_stats(g, thresh, metadata=None, id_col=None):
    if g is None or g.vcount() == 0:
        return {}

    components = g.components()
    comp_sizes = [len(c) for c in components]

    stats = {
        "threshold": thresh,
        "nodes": g.vcount(),
        "mst_edges": g.ecount(),
        "n_components": len(components),
        "largest_component": max(comp_sizes) if comp_sizes else 0,
        "singletons": sum(1 for c in comp_sizes if c == 1)
    }

    annotated_count = count_annotated_nodes(g, metadata, id_col)
    if annotated_count > 0:
        stats["annotated_nodes"] = annotated_count
        stats["annotated_percent"] = round(annotated_count / g.vcount() * 100, 1)

    return stats


def setup_logging(log_file):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stderr)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot Minimum Spanning Tree of Sequence Similarity Network using Plotly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="YAML configuration file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    config_base = infer_config_base(config_path)

    log_file = cfg.get("log_file", "plot_mst.log")
    log_file = resolve_path(log_file, config_base) or Path("plot_mst.log")
    # Use a separate log file for MST
    log_file = Path(str(log_file).replace("plot_ssn.log", "plot_mst.log"))
    setup_logging(log_file)
    logger.info(f"Loaded config from {config_path}")
    print(f"Processing: {config_path}\n")

    # Resolve paths (same logic as plot_ssn.py)
    if "tsv_file" in cfg:
        tsv_file = resolve_path(cfg["tsv_file"], config_base)
        if "output_dir" in cfg and "prefix" in cfg:
            output_dir = resolve_path(cfg["output_dir"], config_base)
            if output_dir:
                output_dir = Path(str(output_dir).rstrip("/"))
            prefix = cfg["prefix"]
            output_base = str(output_dir / "plots" / f"{prefix}_mst")
        else:
            output_path = resolve_path(cfg["output_file"], config_base)
            output_base = str(output_path.with_suffix("")).replace("_ssn", "_mst")
    else:
        prefix = cfg["prefix"]
        output_dir = resolve_path(cfg["output_dir"], config_base)
        if output_dir:
            output_dir = Path(str(output_dir).rstrip("/"))
        tsv_file = output_dir / "search" / f"{prefix}.m8"
        output_base = str(output_dir / "plots" / f"{prefix}_mst")

    thresholds = cfg.get("threshold", [0.3])
    if not isinstance(thresholds, list):
        thresholds = [thresholds]

    annotation_cfg = cfg.get("annotation", {})
    ssn_cfg = cfg.get("ssn", {})

    if "threshold" in ssn_cfg:
        thresholds = ssn_cfg["threshold"]
        if not isinstance(thresholds, list):
            thresholds = [thresholds]

    meta_file = _yn(annotation_cfg.get("meta_file", cfg.get("meta_file")))
    id_col = _yn(annotation_cfg.get("id_col", cfg.get("id_col")))
    color_cols = annotation_cfg.get("color_col", cfg.get("color_col", [None]))
    if not isinstance(color_cols, list):
        color_cols = [color_cols]
    color_cols = [_yn(c) for c in color_cols]
    mmseqs_cluster_file = _yn(annotation_cfg.get("mmseqs_cluster_file", cfg.get("mmseqs_cluster_file")))
    id_mapping_file = _yn(annotation_cfg.get("id_mapping_file", cfg.get("id_mapping_file")))
    node_size = ssn_cfg.get("node_size", cfg.get("node_size", 10))
    exclude_singletons = ssn_cfg.get("exclude_singletons", cfg.get("exclude_singletons", True))
    keep_all_nodes = ssn_cfg.get("keep_all_nodes", cfg.get("keep_all_nodes", False))
    layout_method = ssn_cfg.get("layout", cfg.get("layout", "component_grid"))

    meta_file = resolve_path(meta_file, config_base)
    mmseqs_cluster_file = resolve_path(mmseqs_cluster_file, config_base)
    id_mapping_file = resolve_path(id_mapping_file, config_base)

    logger.info(f"Thresholds: {', '.join(map(str, thresholds))}")
    logger.info(f"Color columns: {', '.join(str(c) for c in color_cols)}")
    logger.info(f"Exclude singleton components: {exclude_singletons}")
    logger.info(f"Keep all nodes: {keep_all_nodes}")
    logger.info(f"Layout: {layout_method}")
    print(f"Thresholds: {', '.join(map(str, thresholds))}")
    print(f"Color columns: {', '.join(str(c) for c in color_cols)}")
    print(f"Exclude singletons: {exclude_singletons}")
    print(f"Keep all nodes: {keep_all_nodes}")

    df_no_self = load_data(tsv_file)
    print()

    meta = None
    if meta_file and id_col:
        with timed_step(f"load annotation table {meta_file}"):
            meta = pd.read_csv(meta_file, sep="\t", dtype={id_col: str})
        logger.info(f"Loaded annotation file: {len(meta)} rows")

    clusters = None
    if mmseqs_cluster_file:
        with timed_step(f"load cluster table {mmseqs_cluster_file}"):
            clusters = pd.read_csv(mmseqs_cluster_file, sep="\t", header=None, names=["rep", "member"])
        logger.info(f"Loaded cluster file: {len(clusters)} rows")

    id_map = None
    if id_mapping_file:
        with timed_step(f"load ID mapping {id_mapping_file}"):
            id_map = pd.read_csv(id_mapping_file, sep="\t", names=["genbank", "uniprot"])
        logger.info(f"Loaded ID mapping: {len(id_map)} pairs")

    color_lookups = {}
    for col in color_cols:
        if meta is not None and id_col and col and col in meta.columns:
            with timed_step(f"build color lookup for {col}"):
                color_lookups[col] = build_color_lookup(meta, id_col, col, clusters, id_map=id_map)
        else:
            color_lookups[col] = None

    all_stats = []

    if keep_all_nodes:
        all_nodes = sorted(set(df_no_self["qseqid"].unique()) | set(df_no_self["sseqid"].unique()))
        logger.info(f"Collected {len(all_nodes)} unique nodes from original data")
    else:
        all_nodes = None

    print(f"\nProcessing {len(thresholds)} threshold(s)...\n")
    for thresh in thresholds:
        thresh_pident = thresh * 100 if thresh < 1 else thresh
        logger.info(f"Threshold: {thresh}")

        with timed_step(f"filter edges for threshold {thresh}"):
            threshold_mask = df_no_self["pident"] >= thresh_pident
            self_hit_mask = df_no_self["qseqid"] == df_no_self["sseqid"]
            df_thresh = df_no_self[threshold_mask & ~self_hit_mask].copy()
        logger.info(f"Edges after threshold filter: {len(df_thresh)}")

        with timed_step(f"build MST graph for threshold {thresh}"):
            g = build_mst_graph(df_thresh, all_nodes=all_nodes if keep_all_nodes else None)

        if g is None or g.vcount() == 0:
            logger.info("No nodes in graph, skipping")
            continue

        if exclude_singletons:
            with timed_step(f"exclude singleton components for threshold {thresh}"):
                g, removed_singletons = exclude_singleton_components(g)
            logger.info(f"Removed singleton components: {removed_singletons}")
            if g is None or g.vcount() == 0:
                logger.info("No nodes remain after excluding singletons, skipping")
                continue

        print(f"  Threshold {thresh}: {g.vcount()} nodes, {g.ecount()} MST edges")

        layout_coords = get_layout(g, layout_method)

        with timed_step(f"compute graph stats for threshold {thresh}"):
            st = graph_stats(g, thresh, metadata=meta, id_col=id_col)
        all_stats.append(st)
        logger.info(f"Stats: {st}")

        for col in color_cols:
            logger.info(f"Color column: {col}")
            color_lookup = color_lookups.get(col)
            if color_lookup is None:
                color_lookup = {name: "Unknown" for name in g.vs["name"]}

            col_label = col if col is not None else "uncolored"
            out_file = f"{output_base}_t{thresh}_{col_label}.html"
            plot_mst(
                g, out_file, col, thresh, tsv_file,
                color_lookup, node_size, layout_coords=layout_coords,
                metadata=meta, id_col=id_col
            )

    stats_file = f"{output_base}_stats.json"
    with timed_step(f"write stats JSON {stats_file}"):
        with open(stats_file, "w") as f:
            json.dump(all_stats, f, indent=2)
    logger.info(f"Stats written to {stats_file}")
    print(f"\nDone! Stats: {stats_file}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
