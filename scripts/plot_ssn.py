#!/usr/bin/env python3
"""
Sequence Similarity Network (SSN) plotter using Plotly.

Reads a YAML config, filters BLAST results by threshold, builds network graphs,
and generates interactive HTML plots.

Usage:
    python plot_ssn.py [config.yaml]
"""

import sys
import argparse
from pathlib import Path
import json

import pandas as pd
import yaml
import igraph as ig
import plotly.graph_objects as go
import plotly.express as px

# Try to import kaleido for PNG export
try:
    import kaleido
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False


def normalize_ids(ids):
    """Normalize sequence IDs to match across different formats."""
    normalized = []
    for idx in ids:
        idx = str(idx)
        # sp|P12345|GENE_ORG  →  P12345
        if "|" in idx:
            parts = idx.split("|")
            if len(parts) >= 2:
                idx = parts[1]
        # plain integer  →  integer.0 (PAZy stores IDs as floats)
        if idx.isdigit():
            idx = f"{idx}.0"
        normalized.append(idx)
    return normalized


def build_color_lookup(meta, id_col, color_col, clusters=None, pazy_ids=None, id_map=None):
    """
    Build a color lookup dictionary for coloring nodes.
    Handles cluster annotation propagation and ID mapping.
    """
    if meta is None or id_col is None or color_col not in meta.columns:
        print(f"  Warning: Missing metadata or color column '{color_col}' not found. All nodes will be 'Unknown'.")
        return {}

    lookup = dict(zip(meta[id_col].astype(str), meta[color_col].astype(str)))

    # Extend with UniProt keys when GenBank→UniProt mapping is supplied
    if id_map is not None and len(id_map) > 0:
        for _, row in id_map.iterrows():
            if str(row["genbank"]) in lookup:
                lookup[str(row["uniprot"])] = lookup[str(row["genbank"])]
        print(f"  Extended lookup with {len(id_map)} UniProt keys")

    # Propagate annotations to cluster representatives via majority vote
    if clusters is not None and len(clusters) > 0:
        cluster_annots = {}
        for _, row in clusters.iterrows():
            rep = str(row["rep"])
            member = str(row["member"])
            member_normalized = normalize_ids([member])[0]

            # Get annotation from member
            annot = None
            if member_normalized in lookup:
                annot = lookup[member_normalized]
            elif member in lookup:
                annot = lookup[member]

            if annot and annot != "":
                if rep not in cluster_annots:
                    cluster_annots[rep] = {}
                cluster_annots[rep][annot] = cluster_annots[rep].get(annot, 0) + 1

        # Assign majority annotation to each rep
        for rep, annot_counts in cluster_annots.items():
            majority = max(annot_counts, key=annot_counts.get)
            lookup[rep] = majority

        print(f"  Propagated cluster annotations for {len(cluster_annots)} representatives")

    return lookup


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def infer_config_base(config_path):
    """Infer the directory that relative paths in the config should use."""
    config_dir = config_path.resolve().parent
    if config_dir.name == "logs" and config_dir.parent.name == "results":
        return config_dir.parent.parent
    return config_dir


def resolve_path(path_value, config_base):
    """Resolve a config path, keeping cwd-relative paths that already work."""
    if not path_value:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return config_base / path


def load_data(tsv_file):
    """Load BLAST-like results TSV file."""
    df = pd.read_csv(
        tsv_file,
        sep="\t",
        names=["qseqid", "sseqid", "pident", "length", "mismatch",
               "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore"],
        dtype={"qseqid": str, "sseqid": str, "pident": float}
    )
    print(f"Loaded {len(df)} rows from {tsv_file}")

    # Remove self-hits (100% identity)
    df_no_self = df[df["pident"] != 100].copy()
    print(f"After removing self-hits: {len(df_no_self)} rows")

    return df_no_self


def create_graph(df_thresh):
    """Create igraph from threshold-filtered dataframe."""
    if len(df_thresh) == 0:
        return None

    # Create edge list
    edges = [(str(row["qseqid"]), str(row["sseqid"])) for _, row in df_thresh.iterrows()]

    # Create graph
    g = ig.Graph.TupleList(edges, directed=False)

    # Add edge weights (pident)
    edge_pident = df_thresh.set_index(["qseqid", "sseqid"])["pident"].to_dict()
    for edge_idx in range(g.ecount()):
        edge = g.es[edge_idx]
        source = g.vs[edge.source]["name"]
        target = g.vs[edge.target]["name"]
        # Try both directions
        pident = edge_pident.get((source, target)) or edge_pident.get((target, source)) or 0
        edge["pident"] = pident

    return g


def get_layout_stress(g):
    """
    Compute stress layout using igraph's layout algorithm.
    Returns x, y coordinates.
    """
    layout = g.layout_fruchterman_reingold(dim=2)
    xs = [coord[0] for coord in layout]
    ys = [coord[1] for coord in layout]
    return xs, ys


def plot_ssn(g, output_file, color_col, thresh, tsv_file,
             color_lookup=None, node_size=5):
    """
    Create an interactive Plotly network visualization.
    """
    if g is None or g.vcount() == 0:
        print(f"  Skipping plot (no nodes)")
        return

    # Get layout
    xs, ys = get_layout_stress(g)

    # Extract node and edge data
    node_names = g.vs["name"]
    node_colors = []
    for name in node_names:
        if color_lookup and name in color_lookup:
            node_colors.append(color_lookup[name])
        else:
            node_colors.append("Unknown")

    # Create edge trace
    edge_x = []
    edge_y = []
    edge_width = []
    edge_hovertext = []

    for edge in g.es:
        source_idx = edge.source
        target_idx = edge.target
        x0, y0 = xs[source_idx], ys[source_idx]
        x1, y1 = xs[target_idx], ys[target_idx]
        pident = edge["pident"] if "pident" in edge.attributes() else 0

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Width proportional to pident
        width = max(0.5, pident / 100 * 2)
        edge_width.extend([width, width, None])

        edge_hovertext.extend([
            f"{node_names[source_idx]} → {node_names[target_idx]}: {pident:.1f}%",
            f"{node_names[source_idx]} → {node_names[target_idx]}: {pident:.1f}%",
            None
        ])

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
        hoverinfo="text",
        hovertext=edge_hovertext,
        showlegend=False
    )

    # Node traces: one fixed-color trace per annotation category.
    # This makes the node glyphs themselves categorical colors and gives Plotly
    # a real legend entry for each value in color_col.
    color_scheme = (
        px.colors.qualitative.Bold
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Dark24
    )
    unique_colors = sorted(c for c in set(node_colors) if c != "Unknown")
    if "Unknown" in node_colors:
        unique_colors.append("Unknown")

    color_map = {
        color: color_scheme[i % len(color_scheme)]
        for i, color in enumerate(unique_colors)
    }
    color_map["Unknown"] = "rgb(128, 128, 128)"
    marker_size = max(float(node_size), 3.0) * 2

    node_traces = []
    for color_value in unique_colors:
        indices = [i for i, value in enumerate(node_colors) if value == color_value]
        node_traces.append(
            go.Scatter(
                x=[xs[i] for i in indices],
                y=[ys[i] for i in indices],
                mode="markers",
                name=str(color_value),
                marker=dict(
                    size=marker_size,
                    color=color_map.get(color_value, "grey"),
                    line=dict(width=0.4, color="white"),
                    opacity=0.9
                ),
                text=[
                    f"{node_names[i]}<br>{color_col}: {node_colors[i]}"
                    for i in indices
                ],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            )
        )

    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)

    # Update layout
    fig.update_layout(
        title={
            "text": f"{Path(tsv_file).name} - Nodes: {g.vcount()}<br>SSN with edges pident >= {thresh}",
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

    # Save HTML (interactive)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)
    print(f"  Saved: {output_file}")
    
    # Save PNG (static) if kaleido available
    if HAS_KALEIDO:
        png_file = str(output_file).replace(".html", ".png")
        try:
            fig.write_image(png_file, width=1200, height=1000, scale=2)
            print(f"  Saved: {png_file}")
        except Exception as e:
            print(f"  Warning: PNG export failed ({e})")
    else:
        png_file = str(output_file).replace(".html", ".png")
        print(f"  Note: Install kaleido to export PNG: pip install kaleido")


def graph_stats(g, thresh):
    """Compute graph statistics."""
    if g is None or g.vcount() == 0:
        return {}

    components = g.components()
    comp_sizes = [len(c) for c in components]

    return {
        "threshold": thresh,
        "nodes": g.vcount(),
        "edges": g.ecount(),
        "n_components": len(components),
        "largest_component": max(comp_sizes) if comp_sizes else 0,
        "singletons": sum(1 for c in comp_sizes if c == 1)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot Sequence Similarity Networks using Plotly",
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
    print(f"Loaded config from {config_path}\n")

    # Extract config parameters
    # Support both formats: direct (tsv_file/output_file) and pipeline (prefix/output_dir)
    if "tsv_file" in cfg:
        # Direct format
        tsv_file = resolve_path(cfg["tsv_file"], config_base)
        output_path = resolve_path(cfg["output_file"], config_base)
        output_base = str(output_path.with_suffix(""))
    else:
        # Pipeline format
        prefix = cfg["prefix"]
        output_dir = resolve_path(cfg["output_dir"], config_base)
        tsv_file = output_dir / "search" / f"{prefix}.m8"
        output_base = str(output_dir / "plots" / f"{prefix}_ssn")

    thresholds = cfg.get("threshold", [0.3])
    if not isinstance(thresholds, list):
        thresholds = [thresholds]
    
    # Handle nested annotation config
    annotation_cfg = cfg.get("annotation", {})
    ssn_cfg = cfg.get("ssn", {})
    
    # Override thresholds from ssn config if available
    if "threshold" in ssn_cfg:
        thresholds = ssn_cfg["threshold"]
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
    
    meta_file = annotation_cfg.get("meta_file", cfg.get("meta_file"))
    id_col = annotation_cfg.get("id_col", cfg.get("id_col"))
    color_cols = annotation_cfg.get("color_col", cfg.get("color_col", [None]))
    if not isinstance(color_cols, list):
        color_cols = [color_cols]
    mmseqs_cluster_file = annotation_cfg.get("mmseqs_cluster_file", cfg.get("mmseqs_cluster_file"))
    id_mapping_file = annotation_cfg.get("id_mapping_file", cfg.get("id_mapping_file"))
    node_size = ssn_cfg.get("node_size", cfg.get("node_size", 10))

    meta_file = resolve_path(meta_file, config_base)
    mmseqs_cluster_file = resolve_path(mmseqs_cluster_file, config_base)
    id_mapping_file = resolve_path(id_mapping_file, config_base)

    print(f"Thresholds: {', '.join(map(str, thresholds))}")
    print(f"Color columns: {', '.join(str(c) for c in color_cols)}\n")

    # Load data once
    df_no_self = load_data(tsv_file)

    # Load metadata
    meta = None
    if meta_file and id_col:
        meta = pd.read_csv(meta_file, sep="\t", dtype={id_col: str})
        print(f"Loaded annotation file: {len(meta)} rows")

    clusters = None
    if mmseqs_cluster_file:
        clusters = pd.read_csv(mmseqs_cluster_file, sep="\t", header=None, names=["rep", "member"])
        print(f"Loaded cluster file: {len(clusters)} rows")

    id_map = None
    if id_mapping_file:
        id_map = pd.read_csv(id_mapping_file, sep="\t", names=["genbank", "uniprot"])
        print(f"Loaded ID mapping: {len(id_map)} pairs\n")

    # Statistics
    all_stats = []

    # Main loop: one graph per threshold, one plot per color column
    for thresh in thresholds:
        # Convert threshold to pident scale (0-100) if given as proportion (0-1)
        thresh_pident = thresh * 100 if thresh < 1 else thresh
        print(f"\n--- Threshold: {thresh} ---")

        df_thresh = df_no_self[df_no_self["pident"] >= thresh_pident].copy()
        print(f"Edges after threshold filter: {len(df_thresh)}")

        g = create_graph(df_thresh)

        if g is None or g.vcount() == 0:
            print(f"No nodes in graph, skipping")
            continue

        print(f"Nodes: {g.vcount()} | Edges: {g.ecount()}")

        # Component statistics
        components = g.components()
        comp_sizes = sorted([len(c) for c in components], reverse=True)
        print(f"Components: {len(components)} | Largest: {comp_sizes[0]}")

        # Collect stats
        st = graph_stats(g, thresh)
        all_stats.append(st)

        # Plot for each color column
        for col in color_cols:
            print(f" Color column: {col}")

            color_lookup = {}
            if meta is not None and id_col and col and col in meta.columns:
                color_lookup = build_color_lookup(meta, id_col, col, clusters, id_map=id_map)
            else:
                # Default: all Unknown
                color_lookup = {name: "Unknown" for name in g.vs["name"]}

            out_file = f"{output_base}_t{thresh}_{col}.html"
            plot_ssn(g, out_file, col, thresh, tsv_file, color_lookup, node_size)

    # Write stats JSON
    stats_file = f"{output_base}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats written to {stats_file}")


if __name__ == "__main__":
    main()
