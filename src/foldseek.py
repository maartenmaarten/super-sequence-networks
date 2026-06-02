"""
Foldseek wrappers for structure-based clustering and all-vs-all search.
Input: directory of PDB / mmCIF files.
pident in the m8 output = TM-score-based identity (0–100).
"""

import logging
import subprocess
import time
from pathlib import Path

_log = logging.getLogger("ssn.foldseek")

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
    pdb_dir: Path,
    output_dir: Path,
    prefix: str,
    min_tmscore: float = 0.5,
    coverage: float = 0.8,
    cov_mode: int = 0,
    alignment_type: int = 2,
    log: logging.Logger = _log,
) -> dict:
    """
    Redundancy reduction with foldseek easy-cluster.

    min_tmscore:    TM-score threshold for cluster membership (0–1)
    alignment_type: 0 = 3Di, 1 = TM-align, 2 = 3Di+AA (default)
    cov_mode:       0 = bidirectional, 1 = query, 2 = target

    Returns:
        cluster_tsv   — two-column TSV (representative \\t member)
        rep_seq_fasta — FASTA of representative sequences
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _run([
        "foldseek", "easy-cluster",
        pdb_dir, output_dir / prefix, output_dir / "tmp",
        "--min-seq-id",      min_tmscore,
        "--cov-mode",        cov_mode,
        "-c",                coverage,
        "--alignment-type",  alignment_type,
    ], log)

    return {
        "cluster_tsv":   output_dir / f"{prefix}_cluster.tsv",
        "rep_seq_fasta": output_dir / f"{prefix}_rep_seq.fasta",
    }


def easy_search(
    pdb_dir: Path,
    output_dir: Path,
    prefix: str,
    alignment_type: int = 2,
    log: logging.Logger = _log,
) -> dict:
    """
    All-vs-all structure similarity search with foldseek easy-search.

    alignment_type: 0 = 3Di, 1 = TM-align, 2 = 3Di+AA (default)

    Returns:
        m8 — path to the tab-separated m8 edge list
             (pident column = TM-score × 100)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    m8_path = output_dir / f"{prefix}.m8"

    _run([
        "foldseek", "easy-search",
        pdb_dir, pdb_dir, m8_path, output_dir / "tmp",
        "--format-output",  _FMT,
        "--alignment-type", alignment_type,
    ], log)

    return {"m8": m8_path}
