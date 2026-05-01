# SSN v2 — Sequence / Structure Similarity Network plotter
# Called by run_ssn.py; reads a YAML config written by the wrapper.
# Produces one PNG per (threshold × color_col) combination and a JSON stats file.

suppressPackageStartupMessages({
  library(igraph)
  library(ggraph)
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(yaml)
  library(jsonlite)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

# ── Helpers ───────────────────────────────────────────────────────────────────

normalize_ids <- function(ids) {
  # sp|P12345|GENE_ORG  →  P12345
  pipe_pat <- "^[a-zA-Z]+\\|([^|]+)\\|.*$"
  ids[grepl(pipe_pat, ids)] <- sub(pipe_pat, "\\1", ids[grepl(pipe_pat, ids)])
  # plain integer  →  integer.0  (PAZy stores IDs as floats)
  int_match <- grepl("^[0-9]+$", ids)
  ids[int_match] <- paste0(ids[int_match], ".0")
  ids
}

build_color_lookup <- function(meta, id_col, color_col, clusters, pazy_ids, id_map = NULL) {
  lookup <- setNames(as.character(meta[[color_col]]), meta[[id_col]])

  # Extend with UniProt keys when a GenBank→UniProt mapping is supplied
  if (!is.null(id_map)) {
    matched <- id_map[id_map$genbank %in% names(lookup), ]
    if (nrow(matched) > 0) {
      lookup[matched$uniprot] <- lookup[matched$genbank]
      cat("  ID mapping: added", nrow(matched), "UniProt keys\n")
    } else {
      cat("  WARNING: id_mapping_file had no overlap with annotation id_col\n")
    }
  }

  # Propagate annotations to cluster representatives via majority vote
  if (!is.null(clusters)) {
    annots <- clusters %>%
      mutate(
        member_key = normalize_ids(.data$member),
        annot = ifelse(
          !is.na(lookup[.data$member_key]),
          lookup[.data$member_key],
          lookup[.data$member]
        ),
        is_pazy = .data$member_key %in% pazy_ids
      ) %>%
      filter(!is.na(.data$annot) & .data$annot != "") %>%
      group_by(.data$rep) %>%
      summarise(
        majority = {
          src <- if (any(.data$is_pazy)) .data$annot[.data$is_pazy] else .data$annot
          tbl <- sort(table(src), decreasing = TRUE)
          if (length(unique(src)) > 1) {
            cat(sprintf(
              "  WARNING: mixed annotations in rep %s — assigning %s\n",
              .data$rep[1], names(tbl)[1]
            ))
          }
          names(tbl)[1]
        },
        .groups = "drop"
      )
    lookup[annots$rep] <- annots$majority
    cat("  Cluster propagation:", nrow(annots), "reps annotated\n")
  }

  lookup
}

assign_colors <- function(layout_coords, g, lookup) {
  keys <- normalize_ids(V(g)$name)
  colors <- ifelse(!is.na(lookup[keys]), lookup[keys], lookup[V(g)$name])
  colors[is.na(colors) | colors == ""] <- "Unknown"
  layout_coords$color_attr <- colors[layout_coords$name]
  layout_coords
}

plot_ssn <- function(layout_coords, output_file, color_col, thresh, tsv_file, node_size = 0.5) {
  uniq <- unique(layout_coords$color_attr)
  pal  <- setNames(rainbow(length(uniq), s = 0.8, v = 0.85), uniq)
  pal["Unknown"] <- "grey50"

  n_nodes <- nrow(layout_coords)
  # Scale output resolution to dataset size
  width  <- if (n_nodes > 5000) 3000 else 1800
  height <- if (n_nodes > 5000) 3000 else 2400

  png(output_file, width = width, height = height, res = 300)
  p <- ggraph(layout_coords) +
    geom_edge_link(aes(alpha = .data$pident), width = 0.3, show.legend = FALSE) +
    geom_node_point(
      aes(filter = .data$color_attr == "Unknown"),
      color = "grey50", size = node_size, alpha = 0.3
    ) +
    geom_node_point(
      aes(filter = .data$color_attr != "Unknown", color = factor(.data$color_attr)),
      size = node_size
    ) +
    geom_node_point(
      aes(filter = .data$highlight),
      color = "red", size = 2, shape = 21, stroke = 2
    ) +
    scale_color_manual(values = pal, na.value = "grey50", name = color_col %||% "") +
    scale_edge_alpha(range = c(0.1, 0.6), guide = "none") +
    theme_void() +
    coord_cartesian(clip = "off") +
    theme(
      legend.position    = c(1.02, 1),
      legend.justification = c("left", "top"),
      legend.background  = element_rect(fill = "white", color = "grey50", linewidth = 0.1),
      plot.margin        = margin(t = 10, r = 150, b = 20, l = 10, unit = "pt"),
      legend.text        = element_text(size = 8),
      legend.key.size    = unit(0.25, "cm"),
      legend.ncol        = 3
    ) +
    labs(
      title    = paste0(basename(tsv_file), "  nodes: ", n_nodes),
      subtitle = paste0("threshold >= ", thresh, "   colour: ", color_col %||% "none")
    )
  print(p)
  dev.off()
  cat("  Saved:", output_file, "\n")
}

graph_stats <- function(g, thresh) {
  comp <- components(g)
  deg  <- degree(g)
  list(
    threshold         = thresh,
    nodes             = vcount(g),
    edges             = ecount(g),
    n_components      = comp$no,
    largest_component = max(comp$csize),
    singletons        = sum(comp$csize == 1),
    avg_degree        = round(mean(deg), 3),
    density           = round(graph.density(g), 6),
    # centroid: node with highest degree in the largest component
    centroid_node     = {
      lc_id  <- which.max(comp$csize)
      lc_verts <- V(g)[comp$membership == lc_id]
      lc_verts$name[which.max(degree(g, lc_verts))]
    }
  )
}

# ── Config ────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
script_dir <- tryCatch(
  dirname(normalizePath(sys.frames()[[1]]$ofile)),
  error = function(e) getwd()
)
config_path <- if (length(args) >= 1) args[1] else file.path(script_dir, "..", "config.yaml")

if (!file.exists(config_path)) stop("Config not found: ", config_path)
cfg <- read_yaml(config_path)
cat("Config:", config_path, "\n")

tsv_file           <- cfg$tsv_file
output_base        <- sub("\\.png$", "", cfg$output_file)
thresholds         <- as.numeric(unlist(cfg$threshold %||% 0))
meta_file          <- cfg$meta_file %||% NULL
id_col             <- cfg$id_col %||% NULL
color_cols         <- as.character(unlist(cfg$color_col %||% list(NULL)))
mmseqs_cluster_file <- cfg$mmseqs_cluster_file %||% NULL
node_size          <- as.numeric(cfg$node_size %||% 0.5)
id_mapping_file    <- cfg$id_mapping_file %||% NULL

cat(sprintf(
  "Thresholds: %s  |  Color cols: %s\n",
  paste(thresholds, collapse = ", "),
  paste(color_cols, collapse = ", ")
))

# ── Load data ─────────────────────────────────────────────────────────────────

m8_cols <- c("qseqid", "sseqid", "pident", "length", "mismatch",
             "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore")
df <- read_tsv(tsv_file, show_col_types = FALSE, col_names = m8_cols)
cat("Loaded", nrow(df), "rows from", tsv_file, "\n")

df_no_self <- df[df$pident != 100, ]
cat("After removing self-hits:", nrow(df_no_self), "\n")

meta      <- NULL
clusters  <- NULL
pazy_ids  <- character(0)
id_map    <- NULL

if (!is.null(meta_file) && !is.null(id_col)) {
  meta <- read_tsv(meta_file, show_col_types = FALSE, col_names = TRUE)
  meta[[id_col]] <- as.character(meta[[id_col]])
  cat("Annotation file:", nrow(meta), "rows\n")
  pazy_ids <- if ("source" %in% names(meta)) {
    as.character(meta[[id_col]][meta$source == "pazy"])
  } else character(0)
}

if (!is.null(mmseqs_cluster_file)) {
  clusters <- read_tsv(mmseqs_cluster_file, show_col_types = FALSE,
                       col_names = c("rep", "member"))
  cat("Cluster file:", nrow(clusters), "rows\n")
}

if (!is.null(id_mapping_file)) {
  id_map <- read_tsv(id_mapping_file, show_col_types = FALSE,
                     col_names = c("genbank", "uniprot"))
  cat("ID mapping:", nrow(id_map), "pairs\n")
}

# Nodes to highlight (edit list as needed)
highlight_ids <- c()

# ── Main loop ─────────────────────────────────────────────────────────────────

all_stats <- list()

for (thresh in thresholds) {
  cat(sprintf("\n--- Threshold: %s ---\n", thresh))

  df_t <- df_no_self[df_no_self$pident >= thresh, ]
  cat("Edges:", nrow(df_t), "\n")

  g <- graph_from_data_frame(df_t, directed = FALSE)
  cat(sprintf("Nodes: %d  Edges: %d\n", vcount(g), ecount(g)))

  # Component summary
  comp   <- components(g)
  sizes  <- sort(table(comp$csize), decreasing = TRUE)
  cat("Components:", comp$no, " | Largest:", max(comp$csize), "\n")
  for (sz in as.integer(names(sizes))) {
    cat(sprintf("  size %d: %d component(s)\n", sz, sizes[[as.character(sz)]]))
  }

  # Collect stats
  st <- graph_stats(g, thresh)
  all_stats <- c(all_stats, list(st))
  cat(sprintf("  Centroid (largest component): %s\n", st$centroid_node))

  # Layout once per threshold
  cat("Computing layout...\n")
  layout_coords <- create_layout(g, layout = "stress")
  layout_coords$highlight <- V(g)$name %in% highlight_ids

  for (col in color_cols) {
    cat(" Color:", col, "\n")
    if (!is.null(meta) && !is.null(id_col) && !is.null(col) && col %in% names(meta)) {
      lookup <- build_color_lookup(meta, id_col, col, clusters, pazy_ids, id_map)
      layout_coords <- assign_colors(layout_coords, g, lookup)
    } else {
      layout_coords$color_attr <- "Unknown"
    }
    out_file <- paste0(output_base, "_t", thresh, "_", col, ".png")
    plot_ssn(layout_coords, out_file, col, thresh, tsv_file, node_size)
  }
}

# Write stats JSON
stats_path <- paste0(output_base, "_stats.json")
write_json(all_stats, stats_path, pretty = TRUE, auto_unbox = TRUE)
cat("\nStats written to", stats_path, "\n")
