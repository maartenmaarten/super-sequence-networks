"""
MMseqs2 wrappers for sequence-based clustering and all-vs-all search.
Input: FASTA file.
pident in the m8 output = % sequence identity (0–100).
"""

import logging
import subprocess
import time
from pathlib import Path

_log = logging.getLogger("ssn.mmseqs")

_FMT = "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"


def _run(cmd: list, log: logging.Logger) -> None:
    log.info("CMD  " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    elapsed = time.time() - t0
    for line in result.stdout.splitlines():
        log.info("     " + line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            log.error("     " + line)
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd[0]}")
    log.info(f"     done in {elapsed:.1f}s")


def easy_cluster(
    fasta: Path,
    output_dir: Path,
    prefix: str,
    min_seq_id: float = 0.8,
    coverage: float = 0.8,
    cov_mode: int = 0,
    log: logging.Logger = _log,
) -> dict:
    """
    Redundancy reduction with mmseqs easy-cluster.

    cov_mode: 0 = bidirectional, 1 = query, 2 = target, 3 = max(query, target)

    Returns:
        cluster_tsv   — two-column TSV (representative \\t member)
        rep_seq_fasta — FASTA of representative sequences
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run([
        "mmseqs", "easy-cluster",
        fasta, output_dir / prefix, output_dir / "tmp",
        "--min-seq-id", min_seq_id,
        "--cov-mode",   cov_mode,
        "-c",           coverage,
    ], log)

    return {
        "cluster_tsv":   output_dir / f"{prefix}_cluster.tsv",
        "rep_seq_fasta": output_dir / f"{prefix}_rep_seq.fasta",
    }


def easy_search(
    fasta: Path,
    output_dir: Path,
    prefix: str,
    evalue: float = 1e-5,
    sensitivity: float = 7.5,
    log: logging.Logger = _log,
) -> dict:
    """
    All-vs-all sequence similarity search with mmseqs easy-search.

    sensitivity: 1 (fast) – 10+ (sensitive); 7.5 is a good default.

    Returns:
        m8 — path to the tab-separated m8 edge list
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    m8_path = output_dir / f"{prefix}.m8"

    _run([
        "mmseqs", "easy-search",
        fasta, fasta, m8_path, output_dir / "tmp",
        "--format-output", _FMT,
        "-e", evalue,
        "-s", sensitivity,
    ], log)

    return {"m8": m8_path}
