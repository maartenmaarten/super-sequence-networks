# SSN v2 — MMseqs / FoldSeek cluster network plotter
# Usage: Rscript plot_clusters.R <cluster_tsv> <output_png> [min_size]
#                                [meta_tsv] [id_col] [color_col] [size_col]
#
# Produces: <output_png>, <output_base>_stats.tsv, <output_base>_stats.json

suppressPackageStartupMessages({
  library(igraph)
  library(ggraph)
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(jsonlite)
})

# ── Args ──────────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript plot_clusters.R <cluster_tsv> <output_png> [min_size]",
      "[meta_tsv] [id_col] [color_col] [size_col]\n")
  quit(status = 1)
}

cluster_tsv <- args[1]
output_png  <- args[2]
min_size    <- if (length(args) >= 3) as.integer(args[3]) else 2L
meta_file   <- if (length(args) >= 4) args[4] else NULL
id_col      <- if (length(args) >= 5) args[5] else NULL
color_col   <- if (length(args) >= 6) args[6] else NULL
size_col    <- if (length(args) >= 7) args[7] else NULL

output_base <- sub("\\.png$", "", output_png)

# ── Load cluster TSV ──────────────────────────────────────────────────────────

clusters <- read_tsv(cluster_tsv, show_col_types = FALSE, col_names = c("rep", "member"))
cat("Loaded", nrow(clusters), "rows from", cluster_tsv, "\n")

# Normalise IDs: sp|ID|... → ID
norm <- function(x) sub("^[a-zA-Z]+\\|([^|]+)\\|.*$", "\\1", x)
clusters <- clusters %>% mutate(rep_key = norm(rep), member_key = norm(member))

# ── Cluster size table ────────────────────────────────────────────────────────

size_tbl <- clusters %>%
  group_by(rep_key) %>%
  summarise(size = n(), .groups = "drop") %>%
  arrange(desc(size))

cat(sprintf(
  "Total clusters: %d  |  Singletons: %d  |  Largest: %d  |  Median size: %.1f\n",
  nrow(size_tbl),
  sum(size_tbl$size == 1),
  max(size_tbl$size),
  median(size_tbl$size)
))

# Filter by minimum cluster size
reps_keep <- size_tbl$rep_key[size_tbl$size >= min_size]
clusters_f <- clusters %>% filter(rep_key %in% reps_keep)
cat(sprintf("After min_size >= %d: %d clusters (%d members)\n",
            min_size, length(reps_keep), nrow(clusters_f)))

# ── Annotation ────────────────────────────────────────────────────────────────

meta_lookup    <- NULL
size_lookup    <- NULL

if (!is.null(meta_file) && !is.null(id_col)) {
  sep  <- if (grepl("\\.csv$", meta_file)) "," else "\t"
  meta <- read_delim(meta_file, delim = sep, show_col_types = FALSE)
  meta[[id_col]] <- as.character(meta[[id_col]])
  cat("Annotation file:", nrow(meta), "rows\n")

  if (!is.null(color_col) && color_col %in% names(meta)) {
    meta_lookup <- setNames(as.character(meta[[color_col]]), meta[[id_col]])
  }
  if (!is.null(size_col) && size_col %in% names(meta)) {
    size_lookup <- setNames(as.numeric(meta[[size_col]]), meta[[id_col]])
  }
}

# Annotation coverage stats
annotated_members <- if (!is.null(meta_lookup)) {
  sum(!is.na(meta_lookup[clusters_f$member_key]) & meta_lookup[clusters_f$member_key] != "")
} else 0L

annotated_reps <- if (!is.null(meta_lookup)) {
  rep_annots <- meta_lookup[reps_keep]
  sum(!is.na(rep_annots) & rep_annots != "")
} else 0L

cat(sprintf(
  "Annotation coverage: members %d/%d (%.1f%%)  reps %d/%d (%.1f%%)\n",
  annotated_members, nrow(clusters_f), 100 * annotated_members / max(nrow(clusters_f), 1),
  annotated_reps,   length(reps_keep), 100 * annotated_reps / max(length(reps_keep), 1)
))

# ── Build graph ───────────────────────────────────────────────────────────────

# Edges: rep → member (exclude self-loops)
edges <- clusters_f %>%
  filter(rep_key != member_key) %>%
  distinct(rep_key, member_key)

g <- graph_from_data_frame(edges, directed = FALSE,
                           vertices = unique(c(clusters_f$rep_key, clusters_f$member_key)))

# Node attributes
V(g)$is_rep  <- V(g)$name %in% reps_keep
V(g)$cluster <- clusters_f$rep_key[match(V(g)$name, clusters_f$member_key)]
V(g)$cluster[is.na(V(g)$cluster)] <- V(g)$name[is.na(V(g)$cluster)]

if (!is.null(meta_lookup)) {
  colors <- meta_lookup[V(g)$name]
  colors[is.na(colors) | colors == ""] <- "Unknown"
  V(g)$color_attr <- colors
} else {
  V(g)$color_attr <- "Unknown"
}

if (!is.null(size_lookup)) {
  sizes <- as.numeric(size_lookup[V(g)$name])
  sizes[is.na(sizes)] <- 1
  V(g)$node_size <- scales::rescale(sizes, to = c(1, 6))
} else {
  V(g)$node_size <- ifelse(V(g)$is_rep, 2, 1)
}

cat(sprintf("Graph: %d nodes, %d edges\n", vcount(g), ecount(g)))

# ── Plot ──────────────────────────────────────────────────────────────────────

uniq_levels <- sort(unique(V(g)$color_attr))
uniq_levels <- c(setdiff(uniq_levels, "Unknown"), "Unknown")
n_col <- sum(uniq_levels != "Unknown")
pal   <- setNames(c(rainbow(n_col, s = 0.8, v = 0.85), "grey50"), uniq_levels)

layout_coords <- create_layout(g, layout = "stress")
layout_coords$color_attr <- V(g)$color_attr[match(layout_coords$name, V(g)$name)]
layout_coords$node_size  <- V(g)$node_size[match(layout_coords$name, V(g)$name)]
layout_coords$is_rep     <- V(g)$is_rep[match(layout_coords$name, V(g)$name)]

n_nodes <- vcount(g)
width   <- if (n_nodes > 3000) 4000 else 3000
height  <- if (n_nodes > 3000) 1500 else 1000

png(output_png, width = width, height = height, res = 300)
p <- ggraph(layout_coords) +
  geom_edge_link(width = 0.2, alpha = 0.4, color = "grey70") +
  geom_node_point(
    aes(filter = color_attr == "Unknown"),
    color = "grey60", size = 1, alpha = 0.4
  ) +
  geom_node_point(
    aes(filter = color_attr != "Unknown", color = factor(color_attr), size = node_size)
  ) +
  geom_node_text(
    aes(filter = is_rep, label = name),
    size = 1.5, repel = TRUE, max.overlaps = 20
  ) +
  scale_color_manual(values = pal, na.value = "grey60", name = color_col %||% "") +
  scale_size_identity() +
  theme_void() +
  coord_cartesian(clip = "off") +
  theme(
    legend.position    = c(1.02, 1),
    legend.justification = c("left", "top"),
    legend.background  = element_rect(fill = "white", color = "grey50", linewidth = 0.1),
    plot.margin        = margin(t = 10, r = 150, b = 10, l = 10, unit = "pt"),
    legend.text        = element_text(size = 7),
    legend.key.size    = unit(0.2, "cm"),
    legend.ncol        = 2
  ) +
  labs(
    title    = paste0(basename(cluster_tsv), "  clusters: ", length(reps_keep)),
    subtitle = paste0("min_size >= ", min_size,
                      if (!is.null(color_col)) paste0("  colour: ", color_col) else "")
  )
print(p)
dev.off()
cat("Saved:", output_png, "\n")

# ── Stats output ──────────────────────────────────────────────────────────────

# Per-cluster stats TSV
cluster_stats <- size_tbl %>%
  mutate(
    annotation = if (!is.null(meta_lookup)) {
      a <- meta_lookup[rep_key]
      ifelse(is.na(a) | a == "", "Unknown", a)
    } else "Unknown"
  )

stats_tsv <- paste0(output_base, "_stats.tsv")
write_tsv(cluster_stats, stats_tsv)
cat("Cluster stats TSV:", stats_tsv, "\n")

# Summary JSON
summary_stats <- list(
  total_clusters    = nrow(size_tbl),
  singletons        = sum(size_tbl$size == 1),
  clusters_plotted  = length(reps_keep),
  largest_cluster   = max(size_tbl$size),
  median_size       = median(size_tbl$size),
  mean_size         = round(mean(size_tbl$size), 2),
  annotation_coverage_members_pct =
    round(100 * annotated_members / max(nrow(clusters_f), 1), 1),
  annotation_coverage_reps_pct    =
    round(100 * annotated_reps / max(length(reps_keep), 1), 1),
  color_col         = color_col %||% NULL
)
stats_json <- paste0(output_base, "_stats.json")
write_json(summary_stats, stats_json, pretty = TRUE, auto_unbox = TRUE)
cat("Cluster stats JSON:", stats_json, "\n")
