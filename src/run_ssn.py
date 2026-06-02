#!/usr/bin/env python3
"""
SSN v2 — pipeline wrapper (Python only).
Usage: python run_ssn.py [--config config.yaml] [--steps all|cluster|search|plot_ssn]

Steps:
  cluster   mmseqs/foldseek easy-cluster  → cluster TSV + rep FASTA
  search    mmseqs/foldseek all-vs-all    → m8 edge list
  plot_ssn  Python: SSN network + stats (Plotly interactive HTML + PNG)
  all       run all three steps in order (default)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from mmseqs   import easy_cluster as mmseqs_cluster, easy_search as mmseqs_search
from foldseek import easy_cluster as foldseek_cluster, easy_search as foldseek_search


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, prefix: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{prefix}_{stamp}.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("ssn")
    log.info(f"Log: {log_path}")
    return log


def log_config(cfg: dict, log: logging.Logger) -> None:
    log.info("=== Config ===")
    for line in json.dumps(cfg, indent=2, default=str).splitlines():
        log.info(line)
    log.info("==============")


# ── Graph plotting runner ────────────────────────────────────────────────────

def run_python_script(script: Path, script_args: list, log: logging.Logger) -> None:
    cmd = ["python", str(script)] + [str(a) for a in script_args]
    log.info("CMD  " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    for line in result.stdout.splitlines():
        log.info("PY   " + line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            log.error("PY   " + line)
        raise RuntimeError(f"Python script exited with code {result.returncode}")
    log.info(f"     done in {elapsed:.1f}s")


# ── Graph statistics (Python-side, optional) ──────────────────────────────────

def compute_graph_stats(m8_path: Path, threshold: float, log: logging.Logger) -> dict:
    try:
        import igraph as ig
    except ImportError:
        log.warning("python-igraph not installed — skipping Python-side graph stats")
        return {}

    edges, weights = [], []
    with open(m8_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, s, pident = parts[0], parts[1], float(parts[2])
            if q == s or pident == 100.0:
                continue
            if pident >= threshold:
                edges.append((q, s))
                weights.append(pident)

    if not edges:
        return {"threshold": threshold, "nodes": 0, "edges": 0}

    g = ig.Graph.TupleList(edges, directed=False)
    comps = g.clusters()
    deg = g.degree()
    return {
        "threshold": threshold,
        "nodes": g.vcount(),
        "edges": g.ecount(),
        "components": len(comps),
        "largest_component": max(comps.sizes()),
        "singletons": sum(1 for s in comps.sizes() if s == 1),
        "avg_degree": round(sum(deg) / len(deg), 3) if deg else 0,
        "avg_pident": round(sum(weights) / len(weights), 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SSN v2 pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["cluster", "search", "plot_ssn", "all"],
        default=["all"],
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    run_all = "all" in args.steps
    steps = set(args.steps)

    input_fasta = Path(cfg["input_fasta"])
    prefix = cfg["prefix"]
    output_dir = Path(cfg["output_dir"])
    scripts_dir = Path(__file__).parent / "scripts"

    log = setup_logging(output_dir / "logs", prefix)
    log_config(cfg, log)

    if not input_fasta.exists():
        log.error(f"Input not found: {input_fasta}")
        sys.exit(1)

    cluster_tsv: Path | None = None
    rep_fasta: Path | None = None
    m8_path: Path | None = None

    # ── Step 1: Cluster ───────────────────────────────────────────────────────
    c = cfg.get("cluster", {})
    if c.get("skip"):
        rep_fasta = input_fasta
        log.info("=== Step 1: Skipped (using full input FASTA for search) ===")
    elif run_all or "cluster" in steps:
        log.info("=== Step 1: Redundancy reduction ===")
        tool = c.get("tool", "mmseqs")
        _cluster = mmseqs_cluster if tool == "mmseqs" else foldseek_cluster
        result = _cluster(
            input_fasta,
            output_dir=output_dir / "cluster",
            prefix=prefix,
            min_seq_id=c.get("min_seq_id", 0.8),
            coverage=c.get("coverage", 0.8),
            cov_mode=c.get("cov_mode", 0),
            log=log,
        )
        cluster_tsv = result["cluster_tsv"]
        rep_fasta = result["rep_seq_fasta"]
        log.info(f"cluster_tsv:   {cluster_tsv}")
        log.info(f"rep_seq_fasta: {rep_fasta}")

    # ── Step 2: All-vs-all search ─────────────────────────────────────────────
    s = cfg.get("search", {})
    if s.get("results_file"):
        m8_path = Path(s["results_file"])
        log.info(f"=== Step 2: Skipped (using existing m8: {m8_path}) ===")
    elif run_all or "search" in steps:
        if rep_fasta is None:
            rep_fasta = output_dir / "cluster" / f"{prefix}_rep_seq.fasta"
        log.info("=== Step 2: All-vs-all search ===")
        tool = s.get("tool", "mmseqs")
        if tool == "mmseqs":
            result = mmseqs_search(
                rep_fasta,
                output_dir=output_dir / "search",
                prefix=prefix,
                evalue=s.get("evalue", 1e-5),
                sensitivity=s.get("sensitivity", 7.5),
                log=log,
            )
        else:
            result = foldseek_search(
                rep_fasta,
                output_dir=output_dir / "search",
                prefix=prefix,
                log=log,
            )
        m8_path = result["m8"]
        log.info(f"m8: {m8_path}")

    # ── Step 3: Plot SSN ──────────────────────────────────────────────────────
    if run_all or "plot_ssn" in steps:
        if m8_path is None:
            m8_path = output_dir / "search" / f"{prefix}.m8"
        log.info("=== Step 3: Plot SSN ===")
        ann = cfg.get("annotation", {})
        ssn = cfg.get("ssn", {})

        thresholds = ssn.get("threshold", [0.3])
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        color_cols = ann.get("color_col", [None])
        if not isinstance(color_cols, list):
            color_cols = [color_cols]

        # Resolve cluster file: prefer annotation override, fall back to step 1 output
        mmseqs_cluster = ann.get("mmseqs_cluster_file") or (str(cluster_tsv) if cluster_tsv else None)

        r_cfg = {
            "tsv_file": str(m8_path),
            "output_dir": str(output_dir),
            "prefix": prefix,
            "threshold": thresholds,
            "node_size": ssn.get("node_size", 0.5),
            "layout": ssn.get("layout", "component_grid"),
            "exclude_singletons": ssn.get("exclude_singletons", True),
            "keep_all_nodes": ssn.get("keep_all_nodes", False),
            "scale_node_size": ssn.get("scale_node_size", False),
            "meta_file": ann.get("meta_file"),
            "id_col": ann.get("id_col"),
            "color_col": color_cols,
            "mmseqs_cluster_file": mmseqs_cluster,
            "id_mapping_file": ann.get("id_mapping_file"),
            "annotation": {
                "exclude": ann.get("exclude"),
            },
        }
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
        r_cfg_path = output_dir / "logs" / f"{prefix}_r_config.yaml"
        with open(r_cfg_path, "w") as fh:
            yaml.dump(r_cfg, fh, default_flow_style=False, allow_unicode=True)
        log.info(f"Config written to {r_cfg_path}")

        run_python_script(scripts_dir / "plot_ssn.py", [str(r_cfg_path)], log)

        # Python-side graph statistics per threshold
        all_stats = []
        for thresh in thresholds:
            stats = compute_graph_stats(m8_path, float(thresh), log)
            if stats:
                log.info(f"Stats t={thresh}: {json.dumps(stats)}")
                all_stats.append(stats)
        if all_stats:
            stats_path = output_dir / "logs" / f"{prefix}_graph_stats.json"
            with open(stats_path, "w") as fh:
                json.dump(all_stats, fh, indent=2)
            log.info(f"Graph stats: {stats_path}")

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
