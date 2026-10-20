import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import functools
    import io
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    from PIL import Image, ImageOps
    from wigglystuff import ObservablePlot

    from iconoscope.dataset import ImageDataset
    from iconoscope.mosaic import layout_mosaic

    # mosaic canvas size in pixels; 16:10, close to typical screens, so the plot
    # fills most of the screen in fullscreen
    CANVAS_WIDTH = 2400
    CANVAS_HEIGHT = 1500
    # max on-screen size of the plot widget, in CSS pixels
    PLOT_MAX_WIDTH = 1400
    PLOT_HEIGHT = 900
    # thumbnail resolution (max width in px); high enough to stay sharp when
    # zoomed into the plot, but each image is embedded in the page
    THUMB_PX = 384
    # space above the plot for the summary line and controls
    PLOT_CHROME = 64
    return (
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        Image,
        ImageDataset,
        ImageOps,
        ObservablePlot,
        PLOT_CHROME,
        PLOT_HEIGHT,
        PLOT_MAX_WIDTH,
        Path,
        THUMB_PX,
        base64,
        functools,
        io,
        layout_mosaic,
        mo,
        np,
        pl,
    )


@app.cell
def _(ImageDataset, Path):
    pgp_dataset_path = Path(
        "/Users/rkoeser/workarea/experiments/20260608-pgp-vislab/pgp.h5"
    )
    img_dataset = ImageDataset(pgp_dataset_path)
    return (img_dataset,)


@app.cell
def _():
    # TODO: summarize info and make readable for users
    # img_dataset.info()
    return


@app.cell
def _(pl):
    # load metadata
    # TODO: this csv changes structure part way through and is not all of the files
    # also -- redundancy?
    image_meta_df = pl.read_csv(
        "/Users/rkoeser/workarea/experiments/20260608-pgp-vislab/image_meta.csv"
    )

    # fix it for now
    image_meta_df = (
        image_meta_df.with_columns(
            real_path=pl.when(pl.col.image_path.str.starts_with("data/pgp-"))
            .then(pl.col.image_path)
            .otherwise(pl.col.shelfmark),
            real_shelfmark=pl.when(pl.col.image_path.str.starts_with("data/pgp-"))
            .then(pl.col.shelfmark)
            .otherwise(pl.col.image_path),
        )
        .drop("image_path", "shelfmark")
        .rename({"real_path": "image_path", "real_shelfmark": "shelfmark"})
    )

    image_meta_df
    return (image_meta_df,)


@app.cell
def _(image_meta_df):
    image_meta_df["keyword"].value_counts()
    return


@app.cell
def _(image_meta_df):
    image_meta_df["type"].value_counts()
    return


@app.cell
def _(pl):
    pgp_meta_df = pl.read_csv(
        "/Users/rkoeser/workarea/github/pgp-metadata/data/documents.csv"
    )
    # split and explode on shelfmark so we get one row per document+shelfmark combination
    pgp_meta_df = (
        pgp_meta_df.with_columns(shelfmark_list=pl.col.shelfmark.str.split(" + "))
        .explode("shelfmark_list")
        .rename({"shelfmark": "shelfmark_orig", "shelfmark_list": "shelfmark"})
    )
    pgp_meta_df
    return (pgp_meta_df,)


@app.cell
def _(image_meta_df, pgp_meta_df):
    # TODO: not sure how this handles shelfmarks associated with multiple documents
    pgp_image_meta_df = image_meta_df.join(pgp_meta_df, on="shelfmark")
    pgp_image_meta_df
    return


@app.cell
def _(image_meta_df, pgp_meta_df, pl):
    # one row per image (image_meta lists some images more than once), with
    # PGP document info attached by shelfmark
    _img = image_meta_df.select("image_path", "shelfmark").unique()
    _docs = (
        pgp_meta_df.with_columns(
            # tags are a comma-separated string per document
            _tags=pl.col.tags.str.split(",").list.eval(
                pl.element().str.strip_chars().filter(pl.element() != "")
            )
        )
        .group_by("shelfmark")
        .agg(
            pl.col.pgpid.unique().sort().alias("pgpids"),
            pl.col.type.drop_nulls().unique().sort().alias("doc_types"),
            # PGP tags combined across documents
            pl.col._tags.explode().drop_nulls().unique().sort().alias("tags"),
        )
        .with_columns(doc_type=pl.col.doc_types.list.join(", "))
    )
    image_info_df = _img.join(_docs, on="shelfmark", how="left")
    image_info_df
    return (image_info_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot images with UMAP (sparse or grid)

    Use the controls to select from the available vision models and pre-processing options and optionally filter the images by document type or keyword.  You can view a sample of the images in UMAP plot or grid.

    Zoom in or pan around on the images to see in more detail; hover over an image to see the shelfmark it belongs to along with associated PGPID(s) and document types.

    Click the full-screen arrows at the top right of the cell to expand the viewer and controls to use your entire screen.
    """)
    return


@app.cell
def _(image_info_df, img_dataset, mo, pl):
    _configs = img_dataset.configs_by_label()
    embedding_select = mo.ui.dropdown(
        options=_configs, value=list(_configs.keys())[0], label="Embedding"
    )
    n_sample = mo.ui.slider(
        20, 1000, value=150, step=10, label="Sample size", show_value=True
    )
    # draw a new random sample with the same settings
    resample_button = mo.ui.button(
        label="New sample", value=0, on_click=lambda v: v + 1
    )
    # optional document type filter; images whose shelfmark is associated with
    # several documents match if any of them has the selected type
    # options only include types present in the image data, labeled with image counts
    _doc_type_counts = (
        image_info_df.select(pl.col.doc_types.explode().alias("doc_type"))
        .drop_nulls()
        .group_by("doc_type")
        .len()
        .sort("doc_type")
    )
    doc_type_select = mo.ui.dropdown(
        options={f"{_t} ({_n:,})": _t for _t, _n in _doc_type_counts.iter_rows()},
        value=None,
        allow_select_none=True,
        label="Document type",
    )
    # optional keyword filter on PGP tags (searchable); images match if any of
    # their documents has any of the selected tags
    _tag_counts = (
        image_info_df.select(pl.col.tags.explode().alias("tag"))
        .drop_nulls()
        .group_by("tag")
        .len()
        .sort("tag")
    )
    tag_select = mo.ui.multiselect(
        options={f"{_t} ({_n:,})": _t for _t, _n in _tag_counts.iter_rows()},
        label="Keywords",
    )
    # displayed above the plot, in the same cell so they're visible in fullscreen
    controls_ui = mo.vstack(
        [
            mo.hstack([embedding_select, n_sample, resample_button], justify="start"),
            mo.hstack([doc_type_select, tag_select], justify="start"),
        ]
    )
    return (
        controls_ui,
        doc_type_select,
        embedding_select,
        n_sample,
        resample_button,
        tag_select,
    )


@app.cell
def _(Image, ImageOps, base64, functools, io):
    # cached per (path, size) so resampling or changing the canvas doesn't
    # reload images; defined in its own cell so the cache survives reruns
    @functools.cache
    def thumbnail_data_uri(path, size):
        """JPEG data URI of the full image scaled to *cover* ``size`` (no cropping)."""
        with Image.open(path) as img:
            _im = ImageOps.cover(img.convert("RGB"), size, Image.LANCZOS)
        _buf = io.BytesIO()
        _im.save(_buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(_buf.getvalue()).decode()

    return (thumbnail_data_uri,)


@app.cell
def _(embedding_select, img_dataset, mo):
    # umap is generated (and saved) on first load for this config
    with mo.status.spinner(f"Loading {embedding_select.selected_key}"):
        umap_df = img_dataset.load_data(embedding_select.value, umap=True)
    return (umap_df,)


@app.cell
def _(image_info_df, mo, pl, umap_df):
    # attach metadata to dataset images; dataset paths are absolute, metadata
    # paths are relative to the experiment dir (data/pgp-images/...)
    umap_meta_df = (
        umap_df.with_columns(
            _rel_path=pl.col.image_path.str.extract(r"(data/pgp-images/.*)$")
        )
        .join(
            image_info_df.rename({"image_path": "_rel_path"}),
            on="_rel_path",
            how="left",
        )
        .drop("_rel_path")
    )
    mo.md(
        f"Metadata found for **{umap_meta_df['shelfmark'].is_not_null().sum()}** of {umap_meta_df.height} images; "
        f"PGP document match for **{umap_meta_df['pgpids'].is_not_null().sum()}**"
    )
    return (umap_meta_df,)


@app.cell
def _(umap_meta_df):
    umap_meta_df
    return


@app.cell
def _(
    doc_type_select,
    n_sample,
    np,
    pl,
    resample_button,
    tag_select,
    umap_meta_df,
):
    # sample after loading, so umap coords cover all images (as in generate_mosaic)
    _df = umap_meta_df
    if doc_type_select.value:
        _df = _df.filter(pl.col.doc_types.list.contains(doc_type_select.value))
    if tag_select.value:
        _df = _df.filter(
            pl.col.tags.list.set_intersection(tag_select.value).list.len() > 0
        )
    # referencing resample_button reruns this cell (new random sample) on click
    resample_button.value
    sample_df = _df.sample(min(n_sample.value, _df.height))
    coords = sample_df["umap"].to_numpy().astype(np.float32)
    paths = sample_df["image_path"].to_list()
    # same rule as generate_mosaic: most common aspect ratio in the data
    img_aspect = float(sample_df["aspect_ratio"].mode()[0])
    _filters = ([doc_type_select.value] if doc_type_select.value else []) + list(
        tag_select.value
    )
    # shown with the controls above the plot
    sample_summary = f"**{len(coords)}** of {_df.height} images{f' ({", ".join(_filters)})' if _filters else ''}, image aspect ratio {img_aspect:.2f}"
    return coords, img_aspect, paths, sample_df, sample_summary


@app.cell
def _(CANVAS_HEIGHT, CANVAS_WIDTH, coords, img_aspect, layout_mosaic, np):
    # exactly the same layout logic as generate_mosaic
    layout = layout_mosaic(coords, CANVAS_WIDTH, CANVAS_HEIGHT, img_aspect)
    grid_cols, grid_rows = layout.grid.cols, layout.grid.rows
    cell_w, cell_h = layout.cell_width, layout.cell_height

    # work in mosaic pixel space so cell and canvas aspect ratios are honored;
    # UMAP coords (0-1) are stretched over the canvas, as assign_to_grid treats them
    scale = np.array([layout.canvas_width, layout.canvas_height], dtype=float)
    dest = np.full_like(coords, np.nan, dtype=float)
    for (_r, _c), _i in layout.assignments.items():
        dest[_i] = ((_c + 0.5) * cell_w, (_r + 0.5) * cell_h)
    placed = ~np.isnan(dest[:, 0])

    # mean distance moved, in units of cell diagonals
    _moved = np.linalg.norm(dest[placed] - coords[placed] * scale, axis=1)
    mean_dist = float(_moved.mean() / np.hypot(cell_w, cell_h))
    return (
        cell_h,
        cell_w,
        dest,
        grid_cols,
        grid_rows,
        layout,
        mean_dist,
        placed,
        scale,
    )


@app.cell
def _(coords, dest, np, placed, sample_df, scale):
    # one record per placed image: start (sx, sy), end (dx, dy), color, label;
    # thumbnails ("img") are added by the plot cell
    _idx = np.flatnonzero(placed)
    _src = coords[_idx] * scale
    _dst = dest[_idx]
    _c = coords[_idx]
    _meta = sample_df.select("shelfmark", "pgpids", "doc_type").to_dicts()

    def _label(i):
        """hover text: whatever document metadata is available"""
        _m = _meta[i]
        _lines = []
        if _m["shelfmark"]:
            _lines.append(_m["shelfmark"])
        if _m["pgpids"]:
            _lines.append("PGPID " + ", ".join(str(_p) for _p in _m["pgpids"]))
        if _m["doc_type"]:
            _lines.append(_m["doc_type"])
        return "\n".join(_lines)

    points = [
        {
            "sx": float(_src[k, 0]),
            "sy": float(_src[k, 1]),
            "dx": float(_dst[k, 0]),
            "dy": float(_dst[k, 1]),
            # 2D color by original UMAP position
            "color": "rgb({},{},{})".format(
                int(255 * _c[k, 0]),
                int(255 * (0.35 + 0.3 * (1 - _c[k, 0]) * _c[k, 1])),
                int(255 * _c[k, 1]),
            ),
            "label": _label(i),
        }
        for k, i in enumerate(_idx)
    ]
    return (points,)


@app.cell
def _(mo):
    # JS shared with ../umap_to_grid.py: builds its own controls and redraws in
    # the browser, so animation never round-trips to Python
    MORPH_JS = (mo.notebook_dir().parent / "umap_to_grid.js").read_text()
    return (MORPH_JS,)


@app.cell
def _(
    MORPH_JS,
    ObservablePlot,
    PLOT_CHROME,
    PLOT_HEIGHT,
    PLOT_MAX_WIDTH,
    THUMB_PX,
    cell_h,
    cell_w,
    controls_ui,
    grid_cols,
    grid_rows,
    img_aspect,
    layout,
    mean_dist,
    mo,
    np,
    paths,
    placed,
    points,
    sample_summary,
    thumbnail_data_uri,
):
    # initial size (and placeholder while loading): fits within PLOT_MAX_WIDTH x
    # PLOT_HEIGHT, preserving canvas aspect; the plot then fills the available width
    _k = min(
        PLOT_MAX_WIDTH / layout.canvas_width,
        (PLOT_HEIGHT - PLOT_CHROME) / layout.canvas_height,
    )
    _plot_w = round(layout.canvas_width * _k)
    _plot_h = round(layout.canvas_height * _k) + PLOT_CHROME

    # thumbnails are loaded here (not in their own cell) so loading progress shows
    # to the right of the controls, with a placeholder the size of the plot, and
    # the page doesn't shift; thumbnail_data_uri caches, so repeats are fast
    _placed_paths = [paths[_i] for _i in np.flatnonzero(placed)]
    _size = (THUMB_PX, max(1, round(THUMB_PX / img_aspect)))
    _thumbs = []
    _errors = []
    for _n, _p in enumerate(_placed_paths):
        if _n % 10 == 0:
            mo.output.replace(
                mo.vstack(
                    [
                        mo.hstack(
                            [
                                controls_ui,
                                mo.md(f"Loading thumbnails… {_n}/{len(_placed_paths)}"),
                            ],
                            justify="start",
                            align="center",
                            gap=2,
                        ),
                        mo.Html(f'<div style="height:{_plot_h + 30}px"></div>'),
                    ]
                )
            )
        try:
            _thumbs.append(thumbnail_data_uri(_p, _size))
        except Exception as _exc:
            _thumbs.append(None)
            _errors.append(f"{_p}: {_exc}")
    _thumb_status = (
        f"Loaded **{sum(_t is not None for _t in _thumbs)}/{len(_thumbs)}** thumbnails."
    )
    if _errors:
        _thumb_status += f" First error: `{_errors[0]}`"

    umap_grid_widget = mo.ui.anywidget(
        ObservablePlot(
            MORPH_JS,
            variables={
                "points": [{**_pt, "img": _t} for _pt, _t in zip(points, _thumbs)],
                "cols": grid_cols,
                "rows": grid_rows,
                "cellW": cell_w,
                "cellH": cell_h,
                "canvasW": layout.canvas_width,
                "canvasH": layout.canvas_height,
                "imgAspect": img_aspect,
                "meanDist": mean_dist,
                "chrome": PLOT_CHROME,
                # fill the available width; width/height below are the initial size
                "fitWidth": True,
            },
            width=_plot_w,
            height=_plot_h,
        )
    )
    mo.vstack(
        [
            controls_ui,
            mo.md(f"{sample_summary} · {_thumb_status}"),
            umap_grid_widget,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Similarity search

    Use "nearest neighbors" search to find visually similar images across all available embeddings.
    """)
    return


if __name__ == "__main__":
    app.run()
