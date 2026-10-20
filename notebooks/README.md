# Notebooks

[marimo](https://marimo.io) notebooks for exploring iconoscope datasets.
Commands below are run from the repository root.

## UMAP → grid notebook

`notebooks/umap_to_grid.py` is a [marimo](https://marimo.io) notebook that
animates how the mosaic layout works: each image moves from its 2D UMAP
position to its assigned grid cell, using the same layout code as
`iconoscope mosaic`. The animation is drawn with
[Observable Plot](https://observablehq.com/plot/) (via
[wigglystuff](https://github.com/koaning/wigglystuff)), and the JavaScript for
it lives in `notebooks/umap_to_grid.js`.

Run it from the repository root:

```console
uv run --with marimo --with wigglystuff marimo edit notebooks/umap_to_grid.py
```

Enter the path to a dataset `.h5` file that has embeddings (from `iconoscope embed`), or leave it blank to use synthetic data. Set the sample size and
mosaic canvas size, and check **Thumbnails** to show images instead of colored
points. In the plot, use ▶ or the slider to run the animation; the controls also
toggle displacement lines and whether thumbnails *cover* (crop to fill) or *fit*
(show whole image) their cells.

### Exporting for the web

The **Export for the web** section at the bottom of the notebook saves the
current grid so it can be embedded elsewhere, e.g. in a blog post:

1. Set the export folder (default `notebooks/export/`, which is git-ignored), a
   name (used in the file name), and a title (shown in a dropdown).
1. Click **Export**. This saves `umap_grid_<name>.json` with the layout and
   inline thumbnails. Exporting again with the same name replaces it.
1. Repeat for other datasets or settings to compare multiple grids.

After each export, two files are regenerated to include every grid in the
folder, with a dropdown to switch between them:

- `umap_grid_observable.js`: a single cell for an
  [Observable](https://observablehq.com) notebook. Attach each
  `umap_grid_*.json` file to the notebook with the same file name, then paste
  the contents of this file into a cell.
- `index.html`: a standalone page that loads the JSON files next to it. Host
  the whole folder, or preview locally with `python -m http.server` in the
  export folder and open <http://localhost:8000/>.

Image labels are exported as file names only, so local paths are not
published. Thumbnails are embedded in the JSON, so file size grows with the
number of images (roughly 10–15 KB per image).

## Nearest neighbors notebook

`notebooks/nearest_neighbors.py` is a [marimo](https://marimo.io) notebook for
comparing embedding models: pick an image and see its nearest neighbors
(cosine similarity) for every model with embeddings in the dataset.

Run it from the repository root:

```console
uv run --with marimo --with altair marimo edit notebooks/nearest_neighbors.py
```

Enter the path to a dataset `.h5` file that has embeddings (from
`iconoscope embed`). Choose an image by index or click **Random image**, and
set the number of neighbors and thumbnail height. The notebook shows:

- one row of neighbor thumbnails per model, labeled with rank and similarity
- UMAP plots centered on the selected image, either combined (neighbors from
  all models overlaid, colored by model) or as separate plots per model;
  background points are sampled to keep the chart size manageable
- a chart of each neighbor's UMAP distance from the selected image by rank

UMAP orientation is arbitrary for each model, so when comparing models in the
combined plot, only distance from the center is meaningful, not direction.

Note that if a model does not yet have UMAP coordinates, they are calculated
and saved to the dataset file when the notebook loads it.

## PGP image explorer notebook (experimental)

> **Experimental:** this notebook is a work in progress, specific to the PGP
> dataset, and may change or move.

`notebooks/experiments/pgp.py` is a [marimo](https://marimo.io) notebook for
exploring [Princeton Geniza Project](https://geniza.princeton.edu/) (PGP)
fragment images by visual similarity. It combines the UMAP → grid view and
nearest neighbor search with PGP document metadata (loaded from
[pgp-metadata](https://github.com/princetongenizalab/pgp-metadata)):

- UMAP plot / grid of a sample of images, filterable by document type,
  keywords, and description; click an image for document details and links,
  or download a mosaic of a filtered set
- similarity search: compare an image's nearest neighbors across all models

The notebook declares its own dependencies (without torch, which is only
needed to generate embeddings), so it can run in a sandbox:

```console
uvx marimo edit --sandbox notebooks/experiments/pgp.py
```

By default it downloads a random sample of 500 PGP images with metadata
(about 28 MB), extracted to `~/.cache/iconoscope/pgp-subset` on first use.
To use other data, set one of these environment variables:

- `PGP_DATA_URL`: url or local path of a data tarball (delete the cache
  directory to switch tarballs)
- `PGP_DATA_DIR`: local directory with `pgp.h5`, `image_meta.csv`, and images
  under `data/pgp-images/`
