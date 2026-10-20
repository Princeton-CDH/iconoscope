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

    # max on-screen size of the plot widget, in CSS pixels
    PLOT_MAX_WIDTH = 1400
    PLOT_HEIGHT = 700
    # space above the plot for the summary line and controls
    PLOT_CHROME = 64
    return (
        Image,
        ImageDataset,
        ImageOps,
        ObservablePlot,
        PLOT_CHROME,
        PLOT_HEIGHT,
        PLOT_MAX_WIDTH,
        Path,
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
    ## umap/mosaic
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
    canvas_w = mo.ui.slider(
        200, 8000, value=2500, step=100, label="Canvas width", show_value=True
    )
    canvas_h = mo.ui.slider(
        200, 8000, value=1000, step=100, label="Canvas height", show_value=True
    )
    show_thumbs = mo.ui.checkbox(value=True, label="Thumbnails")
    # max thumbnail width in px (resolution, not display size; capped at mosaic cell size)
    thumb_px = mo.ui.slider(
        16, 512, value=128, step=16, label="Thumbnail size (px)", show_value=True
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
    # displayed with thumbnail status by the thumbnail cell
    controls_ui = mo.vstack(
        [
            mo.hstack(
                [embedding_select, n_sample, canvas_w, canvas_h], justify="start"
            ),
            mo.hstack(
                [doc_type_select, tag_select, show_thumbs, thumb_px], justify="start"
            ),
        ]
    )
    return (
        canvas_h,
        canvas_w,
        controls_ui,
        doc_type_select,
        embedding_select,
        n_sample,
        show_thumbs,
        tag_select,
        thumb_px,
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
def _(doc_type_select, mo, n_sample, np, pl, tag_select, umap_meta_df):
    # sample after loading, so umap coords cover all images (as in generate_mosaic)
    _df = umap_meta_df
    if doc_type_select.value:
        _df = _df.filter(pl.col.doc_types.list.contains(doc_type_select.value))
    if tag_select.value:
        _df = _df.filter(
            pl.col.tags.list.set_intersection(tag_select.value).list.len() > 0
        )
    sample_df = _df.sample(min(n_sample.value, _df.height))
    coords = sample_df["umap"].to_numpy().astype(np.float32)
    paths = sample_df["image_path"].to_list()
    # same rule as generate_mosaic: most common aspect ratio in the data
    img_aspect = float(sample_df["aspect_ratio"].mode()[0])
    _filters = ([doc_type_select.value] if doc_type_select.value else []) + list(
        tag_select.value
    )
    mo.md(
        f"**{len(coords)}** of {_df.height} images{f' ({", ".join(_filters)})' if _filters else ''}, image aspect ratio {img_aspect:.2f}"
    )
    return coords, img_aspect, paths, sample_df


@app.cell
def _(
    controls_ui,
    img_aspect,
    mo,
    paths,
    show_thumbs,
    thumb_px,
    thumbnail_data_uri,
):
    # thumbnails depend only on the sample, thumb_px, and aspect ratio (not the
    # canvas/grid), so layout changes don't reload them; the browser scales to the cell
    thumbs = None
    _errors = []
    _status = None
    if show_thumbs.value:
        _size = (thumb_px.value, max(1, round(thumb_px.value / img_aspect)))
        thumbs = []
        for _n, _p in enumerate(paths):
            if _n % 50 == 0:
                mo.output.replace(
                    mo.hstack(
                        [controls_ui, mo.md(f"Loading thumbnails… {_n}/{len(paths)}")],
                        justify="start",
                        align="center",
                        gap=2,
                    )
                )
            try:
                thumbs.append(thumbnail_data_uri(_p, _size))
            except Exception as _exc:
                thumbs.append(None)
                _errors.append(f"{_p}: {_exc}")
        _msg = (
            f"Loaded **{sum(t is not None for t in thumbs)}/{len(thumbs)}** thumbnails."
        )
        if _errors:
            _msg += f" First error: `{_errors[0]}`"
        _status = mo.md(_msg)
    mo.hstack([controls_ui, _status], justify="start", align="center", gap=2)
    return (thumbs,)


@app.cell
def _(canvas_h, canvas_w, coords, img_aspect, layout_mosaic, np):
    # exactly the same layout logic as generate_mosaic
    layout = layout_mosaic(coords, canvas_w.value, canvas_h.value, img_aspect)
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
def _(coords, dest, np, placed, sample_df, scale, thumbs):
    # one record per placed image: start (sx, sy), end (dx, dy), color, optional img
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
            "img": thumbs[i] if thumbs else None,
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
    cell_h,
    cell_w,
    grid_cols,
    grid_rows,
    img_aspect,
    layout,
    mean_dist,
    mo,
    points,
):
    # plot fits within PLOT_MAX_WIDTH x PLOT_HEIGHT, preserving canvas aspect
    _k = min(
        PLOT_MAX_WIDTH / layout.canvas_width,
        (PLOT_HEIGHT - PLOT_CHROME) / layout.canvas_height,
    )
    umap_grid_widget = mo.ui.anywidget(
        ObservablePlot(
            MORPH_JS,
            variables={
                "points": points,
                "cols": grid_cols,
                "rows": grid_rows,
                "cellW": cell_w,
                "cellH": cell_h,
                "canvasW": layout.canvas_width,
                "canvasH": layout.canvas_height,
                "imgAspect": img_aspect,
                "meanDist": mean_dist,
                "chrome": PLOT_CHROME,
            },
            width=round(layout.canvas_width * _k),
            height=round(layout.canvas_height * _k) + PLOT_CHROME,
        )
    )
    umap_grid_widget
    return


if __name__ == "__main__":
    app.run()
