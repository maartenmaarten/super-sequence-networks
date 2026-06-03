# SSN v2 — Sequence & Structure Similarity Networks

Build interactive **Sequence Similarity Networks (SSN)** and **Structure Similarity Networks (StSN)** from protein sequences or structures. Nodes are sequences/structures, edges are similarity scores above a threshold. Networks are visualised as interactive HTML plots (Plotly) with optional PNG export.

## Overview

```
FASTA / PDB dir
      │
      ▼
 1. Cluster       mmseqs easy-cluster / foldseek easy-cluster
      │            → removes redundancy, keeps representative sequences [OPTIONAL]
      ▼
 2. All-vs-all    mmseqs easy-search  / foldseek easy-search
      │            → pairwise similarity m8 edge list
      ▼
 3. Plot SSN/MST  plot_ssn.py / plot_mst.py
                   → interactive HTML + PNG per threshold
```

The pipeline supports two modes:

| Mode | Input                 | Tool     | Similarity metric   |
| ---- | --------------------- | -------- | ------------------- |
| SSN  | FASTA file            | mmseqs2  | % sequence identity |
| StSN | PDB / mmCIF directory | foldseek | TM-score × 100      |

---

## Requirements

- Python ≥ 3.10
- [mmseqs2](https://github.com/soedinglab/MMseqs2) (for sequence-based networks)
- [foldseek](https://github.com/steineggerlab/foldseek) (for structure-based networks)
- Python packages: `igraph`, `plotly`, `pandas`, `pyyaml`, `kaleido` (optional, for PNG export)

---

## Quickstart — notebooks

The easiest entry point is the Jupyter notebooks in `notebooks/`:

| Notebook           | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `build_ssn.ipynb`  | Sequence-based SSN from a FASTA file (mmseqs)        |
| `build_stsn.ipynb` | Structure-based StSN from a PDB directory (foldseek) |

Open either notebook and run cells top to bottom. Each major step has an **a/b** alternative — run one, skip the other.

---

## Pipeline — command-line

```bash
python src/run_ssn.py --config config.yaml --steps all
```

`--steps` accepts: `all` (default), `cluster`, `search`, `plot_ssn`.

### config.yaml keys

| Section      | Key                  | Description                                       |
| ------------ | -------------------- | ------------------------------------------------- |
| *(root)*     | `input_fasta`        | Path to FASTA (or PDB directory for foldseek)     |
| *(root)*     | `prefix`             | Output file prefix                                |
| *(root)*     | `output_dir`         | Directory for all outputs                         |
| `cluster`    | `skip`               | Skip redundancy reduction                         |
| `cluster`    | `tool`               | `mmseqs` or `foldseek`                            |
| `cluster`    | `min_seq_id`         | Identity threshold (0–1)                          |
| `cluster`    | `coverage`           | Coverage threshold (0–1)                          |
| `search`     | `tool`               | `mmseqs` or `foldseek`                            |
| `search`     | `sensitivity`        | mmseqs sensitivity (1–10+)                        |
| `search`     | `evalue`             | E-value cutoff (mmseqs only)                      |
| `search`     | `results_file`       | Use a pre-computed m8 file                        |
| `annotation` | `meta_file`          | TSV with metadata for node colouring              |
| `annotation` | `id_col`             | Column in `meta_file` matching sequence IDs       |
| `annotation` | `color_col`          | Column(s) to colour nodes by (one plot each)      |
| `annotation` | `exclude`            | fnmatch filters to drop rows before annotation    |
| `ssn`        | `threshold`          | List of identity thresholds for separate plots    |
| `ssn`        | `layout`             | Graph layout: `component_grid`, `fr`, `lgl`, `kk` |
| `ssn`        | `exclude_singletons` | Drop isolated nodes from plots                    |
| `ssn`        | `keep_all_nodes`     | Include nodes with no above-threshold edges       |

---

## Source files

| File              | Purpose                                             |
| ----------------- | --------------------------------------------------- |
| `src/run_ssn.py`  | Pipeline orchestrator (cluster → search → plot)     |
| `src/plot_ssn.py` | SSN graph building and Plotly visualisation         |
| `src/plot_mst.py` | Minimum Spanning Tree variant of the network        |
| `src/mmseqs.py`   | Thin wrappers around `mmseqs easy-cluster/search`   |
| `src/foldseek.py` | Thin wrappers around `foldseek easy-cluster/search` |

---

## Data

| File                          | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| `data/GH43_0.8-0.8_catdom.fa` | GH43 catalytic domain sequences (clustered 80/80)  |
| `data/GH43_full.tsv`          | Metadata: Family, Domain, Species, GenBank, Source |
| `data/GH43_characterized.fa`  | Biochemically characterised GH43 sequences         |
| `data/GH43.tsv`               | Full GH43 annotation table                         |
| `data/structures/`            | PDB structures for StSN runs                       |

---

## Outputs

All outputs land in `output_dir/` under sub-folders:

```
output_dir/
  cluster/   representative FASTA, cluster TSV
  search/    all-vs-all m8 edge list
  plots/     SSN/MST HTML and PNG files per threshold and colour column
  logs/      run logs
```
