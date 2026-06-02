"""
Wrappers for mmseqs2 and foldseek clustering and all-vs-all search.
Both tools share identical function signatures; switch behaviour with tool="mmseqs"|"foldseek".

For foldseek the *fasta* argument should point to a directory of PDB/mmCIF files
(or a FASTA — foldseek will call ESMFold internally). mmseqs always expects a FASTA.
"""

import logging
import subprocess
import time
from pathlib import Path

_log = logging.getLogger("ssn.cluster")


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
    tool: str = "mmseqs",
    min_seq_id: float = 0.8,
    coverage: float = 0.8,
    cov_mode: int = 0,
    log: logging.Logger = _log,
) -> dict:
    """
    Reduce redundancy with mmseqs or foldseek easy-cluster.

    Returns dict with keys:
      cluster_tsv    — two-column TSV (representative, member)
      rep_seq_fasta  — FASTA of representative sequences
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    out_prefix = output_dir / prefix

    cmd = [
        tool, "easy-cluster",
        fasta,
        out_prefix,
        tmp_dir,
        "--min-seq-id", min_seq_id,
        "--cov-mode", cov_mode,
        "-c", coverage,
    ]
    _run(cmd, log)

    return {
        "cluster_tsv": output_dir / f"{prefix}_cluster.tsv",
        "rep_seq_fasta": output_dir / f"{prefix}_rep_seq.fasta",
    }


def all_vs_all(
    fasta: Path,
    output_dir: Path,
    prefix: str,
    tool: str = "mmseqs",
    evalue: float = 1e-5,
    sensitivity: float = 7.5,
    log: logging.Logger = _log,
) -> dict:
    """
    All-vs-all search producing an m8 edge list.

    mmseqs  → sequence similarity  (pident column = % sequence identity)
    foldseek → structure similarity (pident column = TMscore-based identity)

    Returns dict with key:
      m8  — path to the m8 edge list
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    m8_path = output_dir / f"{prefix}.m8"

    fmt = "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"

    if tool == "mmseqs":
        cmd = [
            "mmseqs", "easy-search",
            fasta, fasta,
            m8_path,
            tmp_dir,
            "--format-output", fmt,
            "-e", evalue,
            "-s", sensitivity,
        ]
    elif tool == "foldseek":
        cmd = [
            "foldseek", "easy-search",
            fasta, fasta,
            m8_path,
            tmp_dir,
            "--format-output", fmt,
        ]
    else:
        raise ValueError(f"Unknown tool: {tool!r}. Use 'mmseqs' or 'foldseek'.")

    _run(cmd, log)
    return {"m8": m8_path}
