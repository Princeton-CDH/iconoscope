# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anywidget",
#     "h5py",
#     "iconoscope @ git+https://github.com/rlskoeser/iconoscope@feature/pgp-notebook",
#     "marimo>=0.25.1",
#     "numpy",
#     "piffle",
#     "pillow",
#     "polars",
#     "traitlets",
#     "wigglystuff>=0.5.36",
# ]
#
# [tool.uv]
# # torch etc. are only needed to generate embeddings; skip them here
# override-dependencies = [
#     "accelerate; sys_platform == 'never'",
#     "torch; sys_platform == 'never'",
#     "torchvision; sys_platform == 'never'",
#     "transformers; sys_platform == 'never'",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="PGP image explorer")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PGP image explorer

    Explore images of manuscript fragments from the
    [Princeton Geniza Project](https://geniza.princeton.edu/) (PGP) by visual
    similarity, using image embeddings from several vision models
    (e.g. CLIP, SigLIP, DINOv2/v3, ResNet), some with different image
    preprocessing options (center crop, grayscale).

    **Data:** fragment images retrieved via IIIF, with embeddings and 2D
    [UMAP](https://umap-learn.readthedocs.io/) projections precomputed with
    [iconoscope](https://github.com/rlskoeser/iconoscope) and stored in an HDF5 file.
    Images are linked by shelfmark to PGP document metadata (document type,
    tags, description), loaded from
    [pgp-metadata](https://github.com/princetongenizalab/pgp-metadata).

    **What you can do:**

    - **UMAP plot / grid**: view a sample of images positioned by visual
      similarity, then animate them into a grid; filter by document type,
      keywords, or description; click an image for document details and links;
      download a mosaic of a filtered set
    - **Similarity search**: pick a starting image (random, filtered, or from
      the plot) and compare its nearest neighbors across all models side by side
    """)
    return


@app.cell(hide_code=True)
def _(mo, use_iiif):
    # one image source setting for both the UMAP plot and similarity search
    mo.vstack(
        [
            use_iiif,
            mo.md(
                "<small>Applies to both the UMAP plot and similarity search. "
                "Images whose local file is missing load from IIIF "
                "regardless.</small>"
            ).style({"opacity": "0.7"}),
        ]
    )
    return


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
def _(
    controls_ui,
    document_links_md,
    image_src,
    mo,
    mosaic_download,
    nn_image_df,
    nn_set_start,
    nn_start_token,
    pl,
    sample_df,
    sample_summary,
    umap_grid_widget,
    umap_thumb_status,
):
    # details + external links for the image clicked in the UMAP plot
    _sel = umap_grid_widget.value["selected"]
    if _sel < 0:
        umap_panel = mo.md("*Click an image for details and links.*").style(
            {"font-size": "0.85em", "opacity": "0.7"}
        )
    else:
        _row = sample_df.row(_sel, named=True)
        # same image in the similarity search, to start a search from it
        _nn_idx = nn_image_df.filter(pl.col.image_path == _row["image_path"])["idx"]
        _search = mo.ui.button(
            label="Find similar images",
            disabled=_nn_idx.is_empty(),
            on_click=lambda _: nn_set_start((int(_nn_idx[0]), nn_start_token)),
        )
        umap_panel = mo.vstack(
            [
                mo.image(
                    image_src(_row, 600),
                    style={"max-height": "300px", "max-width": "100%", "width": "auto"},
                ),
                mo.md(document_links_md(_row)).style({"font-size": "0.85em"}),
                _search,
                mo.md(
                    "<small>sets the starting image in Similarity search, below</small>"
                ).style({"opacity": "0.7"}),
            ],
            align="start",
        )
    # controls, plot, and panel in one output, so all are visible in fullscreen
    mo.vstack(
        [
            controls_ui,
            mo.hstack(
                [mo.md(f"{sample_summary} · {umap_thumb_status}"), mosaic_download],
                justify="start",
                align="center",
                gap=2,
            ),
            mo.hstack(
                [
                    mo.vstack([umap_grid_widget]).style(
                        {"flex": "1 1 0", "min-width": "0"}
                    ),
                    mo.vstack([umap_panel]).style({"flex": "none", "width": "320px"}),
                ],
                justify="start",
                align="start",
                gap=1,
            ),
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


@app.cell
def _(
    document_links_md,
    image_src,
    mo,
    nn_candidates,
    nn_description_search,
    nn_doc_type_select,
    nn_grid,
    nn_image_df,
    nn_k,
    nn_model_picker,
    nn_neighbors,
    nn_panel,
    nn_query_idx,
    nn_random_button,
    nn_selected_models,
    nn_shelfmark_search,
    nn_show_captions,
    nn_tag_select,
    nn_thumb_size,
):
    # filters and controls, then query image (with document metadata below)
    # and model picker next to the neighbors, in one output, so everything is visible in fullscreen
    _filter_ui = mo.vstack(
        [
            mo.md(
                "**Choose a starting image** — filters narrow the images "
                "*Random image* picks from; neighbors are always searched "
                "across the full dataset."
            ),
            mo.hstack(
                [
                    nn_doc_type_select,
                    nn_tag_select,
                    nn_description_search,
                    nn_shelfmark_search,
                ],
                justify="start",
            ),
            mo.md(
                f"**{nn_candidates.height:,}** of {nn_image_df.height:,} images "
                "match the filters"
            ),
        ]
    )
    # how to read a thumbnail: a mock thumbnail with the same caption and
    # badge styles as the results grid
    _legend = mo.Html(
        """
        <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.75rem; line-height:1.3;">
          <div style="position:relative; width:64px; height:44px; flex:none;
              background:repeating-linear-gradient(45deg, #ccc 0 6px, #ddd 6px 12px);
              border-radius:2px;">
            <span style="position:absolute; left:0; bottom:0; padding:0 3px; font-size:0.6rem;
                color:white; background:rgba(0,0,0,0.55);">#2 · 0.913</span>
            <span style="position:absolute; right:2px; top:2px; padding:0 4px; border-radius:8px;
                font-size:0.65rem; font-weight:600; color:white; background:#ff7f0e;">×3</span>
          </div>
          <div>
            <b>#2</b> rank · <b>0.913</b> cosine similarity to the starting image<br>
            <span style="color:#ff7f0e; font-weight:600;">×3</span> returned by 3 of the shown models;
            hover to outline every copy; click for details and links
          </div>
        </div>
        """
    )
    if nn_query_idx is None:
        _query = mo.callout(mo.md("No images match the filters."), kind="warn")
    else:
        _query = mo.vstack(
            [
                mo.md("**Starting image**"),
                mo.image(
                    image_src(nn_image_df.row(nn_query_idx, named=True), 500),
                    style={
                        "height": "250px",
                        "width": "auto",
                        "max-width": "300px",
                        "object-fit": "contain",
                    },
                ),
                # same document details + links as the results panel
                mo.md(
                    document_links_md(nn_image_df.row(nn_query_idx, named=True))
                ).style({"font-size": "0.85em"}),
            ],
            align="start",
        )
    if nn_query_idx is not None and not nn_selected_models:
        _results = mo.callout(mo.md("Select at least one model."), kind="warn")
    else:
        _results = nn_grid
    mo.vstack(
        [
            _filter_ui,
            # display controls in one row, to leave vertical space for results
            mo.hstack(
                [
                    nn_random_button,
                    nn_k,
                    nn_thumb_size,
                    nn_show_captions,
                    _legend,
                ],
                justify="start",
                align="center",
                gap=2,
            ),
            mo.md(
                f"Nearest neighbors across all **{nn_image_df.height:,}** images, "
                f"for {len(nn_neighbors)} model configuration"
                f"{'' if len(nn_neighbors) == 1 else 's'}"
            ),
            # starting image and model picker to the left of the neighbor rows
            mo.hstack(
                [
                    mo.vstack(
                        [
                            _query,
                            mo.md(
                                "**Models** — check to display; drag or use arrows to reorder"
                            ),
                            nn_model_picker,
                        ],
                        align="start",
                        # keep its natural width (no shrinking under the results)
                    ).style({"flex": "none", "max-width": "420px"}),
                    # results take the remaining width; rows scroll individually
                    mo.vstack([_results]).style({"flex": "1 1 0", "min-width": "0"}),
                    # details + links for the clicked thumbnail
                    mo.vstack([nn_panel]).style({"flex": "none", "width": "320px"}),
                ],
                justify="start",
                align="start",
                gap=1,
            ),
        ]
    )
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
    # options shared with the similarity search filters
    doc_type_options = {f"{_t} ({_n:,})": _t for _t, _n in _doc_type_counts.iter_rows()}
    doc_type_select = mo.ui.dropdown(
        options=doc_type_options,
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
    tag_options = {f"{_t} ({_n:,})": _t for _t, _n in _tag_counts.iter_rows()}
    tag_select = mo.ui.multiselect(
        options=tag_options,
        label="Keywords",
    )
    # case-insensitive text search on PGP document descriptions
    description_search = mo.ui.text(
        label="Description", placeholder="search descriptions"
    )
    # displayed above the plot, in the same cell so they're visible in fullscreen
    controls_ui = mo.vstack(
        [
            mo.hstack([embedding_select, n_sample, resample_button], justify="start"),
            mo.hstack(
                [doc_type_select, tag_select, description_search], justify="start"
            ),
        ]
    )
    return (
        controls_ui,
        description_search,
        doc_type_options,
        doc_type_select,
        embedding_select,
        n_sample,
        resample_button,
        tag_options,
        tag_select,
    )


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
def _(
    description_search,
    doc_type_select,
    filter_images,
    n_sample,
    np,
    resample_button,
    tag_select,
    umap_meta_df,
):
    # sample after loading, so umap coords cover all images (as in generate_mosaic)
    # images matching the current filters (all of them, not just the sample)
    filtered_df = filter_images(
        umap_meta_df,
        doc_type=doc_type_select.value,
        tags=tag_select.value,
        description=description_search.value,
    )
    # referencing resample_button reruns this cell (new random sample) on click
    resample_button.value
    sample_df = filtered_df.sample(min(n_sample.value, filtered_df.height))
    coords = sample_df["umap"].to_numpy().astype(np.float32)
    paths = sample_df["image_path"].to_list()
    # same rule as generate_mosaic: most common aspect ratio in the data
    img_aspect = float(sample_df["aspect_ratio"].mode()[0])
    _filters = (
        ([doc_type_select.value] if doc_type_select.value else [])
        + list(tag_select.value)
        + (
            [f'"{description_search.value.strip()}"']
            if description_search.value.strip()
            else []
        )
    )
    # shown with the controls above the plot
    sample_summary = f"**{len(coords)}** of {filtered_df.height} images{f' ({", ".join(_filters)})' if _filters else ''}, image aspect ratio {img_aspect:.2f}"
    return coords, filtered_df, img_aspect, paths, sample_df, sample_summary


@app.cell
def _(
    MOSAIC_CELL_WIDTH,
    MOSAIC_MAX_WIDTH,
    doc_type_select,
    embedding_select,
    filtered_df,
    io,
    mo,
    np,
    re,
    render_mosaic,
    tag_select,
):
    # download a grid mosaic of every image in the current filtered set, using
    # the same layout and rendering as the mosaic command; generated on click
    def _mosaic_jpeg():
        _aspect = float(filtered_df["aspect_ratio"].mode()[0])
        # canvas size for the target cell size, 16:10, capped at MOSAIC_MAX_WIDTH
        _cell_area = MOSAIC_CELL_WIDTH * (MOSAIC_CELL_WIDTH / _aspect)
        _width = min(
            MOSAIC_MAX_WIDTH, round((filtered_df.height * _cell_area * 16 / 10) ** 0.5)
        )
        _canvas = render_mosaic(
            filtered_df["image_path"].to_list(),
            filtered_df["umap"].to_numpy().astype(np.float32),
            _aspect,
            _width,
            round(_width * 10 / 16),
        )
        _buf = io.BytesIO()
        _canvas.save(_buf, format="JPEG", quality=90)
        return _buf.getvalue()

    def _mosaic_filename():
        _parts = ["mosaic", embedding_select.selected_key]
        if doc_type_select.value:
            _parts.append(doc_type_select.value)
        _parts.extend(tag_select.value)
        _parts.append(str(filtered_df.height))
        return (
            "_".join(re.sub(r"[^A-Za-z0-9.=-]+", "-", _p).strip("-") for _p in _parts)
            + ".jpg"
        )

    # only for a filtered set; the full dataset is too large for one image
    _filtered = bool(doc_type_select.value or tag_select.value)
    _download_button = mo.download(
        data=_mosaic_jpeg,
        filename=_mosaic_filename,
        mimetype="image/jpeg",
        disabled=not _filtered,
        label=f"Download mosaic ({filtered_df.height:,} images)",
    )
    # button, with a note beside it when disabled
    mosaic_download = mo.hstack(
        [_download_button]
        + (
            []
            if _filtered
            else [mo.md("*Filter by document type or keyword to download a mosaic*")]
        ),
        justify="start",
        align="center",
        gap=0.5,
    )
    return (mosaic_download,)


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
    # one record per placed image: start (sx, sy), end (dx, dy), color, label, id;
    # thumbnails ("img") are added by the plot cell
    _idx = np.flatnonzero(placed)
    _src = coords[_idx] * scale
    _dst = dest[_idx]
    _c = coords[_idx]
    _meta = sample_df.select("shelfmark", "pgpids", "doc_type", "iiif_image").to_dicts()

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
            "label": image_label(_meta[i]),
            # row in sample_df, sent back when the point is clicked
            "id": int(i),
        }
        for k, i in enumerate(_idx)
    ]
    return (points,)


@app.cell
def _(
    MORPH_JS,
    PLOT_CHROME,
    PLOT_HEIGHT,
    PLOT_MAX_WIDTH,
    SelectablePlot,
    THUMB_PX,
    cell_h,
    cell_w,
    grid_cols,
    grid_rows,
    iiif_url,
    img_aspect,
    layout,
    mean_dist,
    mo,
    np,
    paths,
    placed,
    points,
    prefer_iiif,
    sample_df,
    thumbnail_data_uri,
):
    # initial size: fits within PLOT_MAX_WIDTH x
    # PLOT_HEIGHT, preserving canvas aspect; the plot then fills the available width
    _k = min(
        PLOT_MAX_WIDTH / layout.canvas_width,
        (PLOT_HEIGHT - PLOT_CHROME) / layout.canvas_height,
    )
    _plot_w = round(layout.canvas_width * _k)
    _plot_h = round(layout.canvas_height * _k) + PLOT_CHROME

    # thumbnails, with a progress bar that's removed when done (the plot is
    # displayed with the details panel, in the next cell); thumbnail_data_uri
    # caches, so repeats are fast
    _placed_paths = [paths[_i] for _i in np.flatnonzero(placed)]
    _size = (THUMB_PX, max(1, round(THUMB_PX / img_aspect)))
    _thumbs = []
    _errors = []
    _placed_iiif = sample_df["iiif_image"].gather(np.flatnonzero(placed)).to_list()
    for _p, _iiif in mo.status.progress_bar(
        list(zip(_placed_paths, _placed_iiif)),
        title="Loading thumbnails",
        remove_on_exit=True,
    ):
        try:
            if prefer_iiif(_p, _iiif):
                # loaded by the browser; the plot crops to the cell
                _thumbs.append(iiif_url(_iiif, width=THUMB_PX))
            else:
                _thumbs.append(thumbnail_data_uri(_p, _size))
        except Exception as _exc:
            _thumbs.append(None)
            _errors.append(f"{_p}: {_exc}")
    umap_thumb_status = (
        f"Loaded **{sum(_t is not None for _t in _thumbs)}/{len(_thumbs)}** thumbnails."
    )
    if _errors:
        umap_thumb_status += f" First error: `{_errors[0]}`"

    umap_grid_widget = mo.ui.anywidget(
        # click a point for details and links (panel to the right)
        SelectablePlot(
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
    # displayed with the details panel, in the next cell
    return umap_grid_widget, umap_thumb_status


@app.cell
def _(doc_type_options, img_dataset, mo, tag_options):
    # all model + preprocessing configs with stored features, keyed by label
    nn_models = img_dataset.configs_by_label()
    nn_random_button = mo.ui.button(
        label="Random image", value=0, on_click=lambda v: v + 1
    )
    nn_k = mo.ui.slider(3, 30, value=10, label="Neighbors (k)", show_value=True)
    nn_thumb_size = mo.ui.slider(
        60, 250, value=120, step=10, label="Thumbnail height", show_value=True
    )
    nn_show_captions = mo.ui.checkbox(value=True, label="Show rank · similarity")
    # filters for choosing the query image (same as the umap filters, plus shelfmark)
    nn_doc_type_select = mo.ui.dropdown(
        options=doc_type_options,
        value=None,
        allow_select_none=True,
        label="Document type",
    )
    nn_tag_select = mo.ui.multiselect(options=tag_options, label="Keywords")
    nn_description_search = mo.ui.text(
        label="Description", placeholder="search descriptions"
    )
    nn_shelfmark_search = mo.ui.text(label="Shelfmark", placeholder="e.g. T-S Ar.52")
    return (
        nn_description_search,
        nn_doc_type_select,
        nn_k,
        nn_models,
        nn_random_button,
        nn_shelfmark_search,
        nn_show_captions,
        nn_tag_select,
        nn_thumb_size,
    )


@app.cell(hide_code=True)
def _(ModelPicker, img_dataset, mo, nn_models):
    # model checklist: check to show, drag (or use arrows) to reorder
    _info = img_dataset.info()["models"]

    def _model_item(label, config):
        _data = _info.get(label, {})
        _meta = [f"{config.model.dimensions}d"]
        if "umap" in _data:
            _meta.append("umap")
        if "cluster" in _data:
            _meta.append(f"k={_data['cluster']['k']}")
        return {
            "model": config.model.name,
            # one badge per set option
            "options": config.option_labels,
            "meta": " · ".join(_meta),
        }

    nn_model_picker = mo.ui.anywidget(
        ModelPicker(
            items={_l: _model_item(_l, _c) for _l, _c in nn_models.items()},
            order=list(nn_models),
            selected=list(nn_models),
        )
    )
    return (nn_model_picker,)


@app.cell
def _(image_info_df, img_dataset, mo, nn_models, np, pl):
    # load paths once, then L2-normalized features for each model, so
    # dot product == cosine similarity
    # with document metadata (as for umap_meta_df) for hover labels
    nn_image_df = (
        img_dataset.get_image_data()
        .with_row_index("idx")
        .with_columns(_rel_path=pl.col.image_path.str.extract(r"(data/pgp-images/.*)$"))
        .join(
            image_info_df.rename({"image_path": "_rel_path"}),
            on="_rel_path",
            how="left",
        )
        .drop("_rel_path")
        .sort("idx")
    )
    nn_features = {}
    with mo.status.progress_bar(
        total=len(nn_models), title="Loading embeddings", remove_on_exit=True
    ) as _bar:
        for _label, _config in nn_models.items():
            _feats = (
                img_dataset.load_data(config=_config, features=True)["features"]
                .to_numpy()
                .astype(np.float32)
            )
            _feats /= np.linalg.norm(_feats, axis=1, keepdims=True).clip(min=1e-12)
            nn_features[_label] = _feats
            _bar.update()
    return nn_features, nn_image_df


@app.cell
def _(mo):
    # starting image chosen from the results panel, stored with the random
    # button count + filters at the time, so a new random pick or filter
    # change replaces it
    nn_get_start, nn_set_start = mo.state(None)
    return nn_get_start, nn_set_start


@app.cell
def _(
    filter_images,
    nn_description_search,
    nn_doc_type_select,
    nn_get_start,
    nn_image_df,
    nn_random_button,
    nn_shelfmark_search,
    nn_tag_select,
    np,
):
    # images matching the filters; a random one is the query, and a new one is
    # picked when the button is clicked
    nn_candidates = filter_images(
        nn_image_df,
        doc_type=nn_doc_type_select.value,
        tags=nn_tag_select.value,
        description=nn_description_search.value,
        shelfmark=nn_shelfmark_search.value,
    )
    # identifies the current random pick + filters; a starting image chosen
    # from the results panel only applies while this is unchanged
    nn_start_token = (
        nn_random_button.value,
        nn_doc_type_select.value,
        tuple(nn_tag_select.value),
        nn_description_search.value,
        nn_shelfmark_search.value,
    )
    _chosen = nn_get_start()
    # no query (None) when nothing matches; the results cell shows a warning,
    # so the filters stay visible
    if _chosen is not None and _chosen[1] == nn_start_token:
        nn_query_idx = _chosen[0]
    elif nn_candidates.height:
        nn_query_idx = int(
            nn_candidates["idx"][int(np.random.randint(nn_candidates.height))]
        )
    else:
        nn_query_idx = None
    return nn_candidates, nn_query_idx, nn_start_token


@app.cell
def _(nn_features, nn_image_df, nn_k, nn_model_picker, nn_query_idx, np, pl):
    # nearest neighbors per selected model (excluding the query image itself)
    nn_neighbors = {}
    # empty when there's no query image
    # checked models, in display order
    nn_selected_models = [
        _l
        for _l in nn_model_picker.value["order"]
        if _l in nn_model_picker.value["selected"]
    ]
    for _name in nn_selected_models if nn_query_idx is not None else []:
        _feats = nn_features[_name]
        _sims = _feats @ _feats[nn_query_idx]
        _sims[nn_query_idx] = -np.inf
        _k = nn_k.value
        _top = np.argpartition(-_sims, _k)[:_k]
        _top = _top[np.argsort(-_sims[_top])]
        nn_neighbors[_name] = pl.DataFrame(
            {"idx": _top, "rank": np.arange(1, _k + 1), "similarity": _sims[_top]}
        ).join(
            nn_image_df.select(
                "idx", "image_path", "shelfmark", "pgpids", "doc_type", "iiif_image"
            ),
            on="idx",
            how="left",
        )
    return nn_neighbors, nn_selected_models


@app.cell
def _(
    NeighborGrid,
    iiif_url,
    image_data_uri,
    mo,
    nn_models,
    nn_neighbors,
    nn_show_captions,
    nn_thumb_size,
    prefer_iiif,
):
    # results grid: one row per model config; images returned by more than
    # one model get a count badge (and are outlined together on hover)
    _returned_by = {}
    for _name, _df in nn_neighbors.items():
        for _idx in _df["idx"]:
            _returned_by.setdefault(_idx, []).append(_name)
    # thumbnails at the max slider height, scaled down in the browser, so
    # changing the height doesn't reload images
    _src_h = 250
    nn_grid = mo.ui.anywidget(
        NeighborGrid(
            rows=[
                {
                    "model": nn_models[_name].model.name,
                    "options": nn_models[_name].option_labels,
                    "thumbs": [
                        {
                            "idx": _r["idx"],
                            "src": (
                                iiif_url(_r["iiif_image"], height=_src_h)
                                if prefer_iiif(_r["image_path"], _r["iiif_image"])
                                else image_data_uri(_r["image_path"], _src_h)
                            ),
                            "rank": _r["rank"],
                            "similarity": float(_r["similarity"]),
                            "count": len(_returned_by[_r["idx"]]),
                            "title": image_label(_r),
                        }
                        for _r in _df.iter_rows(named=True)
                    ],
                }
                for _name, _df in nn_neighbors.items()
            ],
            thumb_height=nn_thumb_size.value,
            show_captions=nn_show_captions.value,
        )
    )
    return (nn_grid,)


@app.cell
def _(
    document_links_md,
    image_src,
    mo,
    nn_grid,
    nn_image_df,
    nn_neighbors,
    nn_set_start,
    nn_start_token,
    pl,
):
    # details + external links for the image clicked in the results grid
    _sel = nn_grid.value["selected"]
    if _sel < 0:
        nn_panel = mo.md("*Click a thumbnail for details and links.*").style(
            {"font-size": "0.85em", "opacity": "0.7"}
        )
    else:
        _row = nn_image_df.row(_sel, named=True)
        # rank of this image for each shown model
        _ranks = [
            f"{_name} #{_df.filter(pl.col.idx == _sel)['rank'][0]}"
            for _name, _df in nn_neighbors.items()
            if _sel in _df["idx"]
        ]
        _use = mo.ui.button(
            label="Use as starting image",
            on_click=lambda _: nn_set_start((_sel, nn_start_token)),
        )
        nn_panel = mo.vstack(
            [
                mo.image(
                    image_src(_row, 600),
                    style={"max-height": "300px", "max-width": "100%", "width": "auto"},
                ),
                mo.md(
                    document_links_md(_row)
                    + "\n\n"
                    + f"<small>Ranked by: {', '.join(_ranks)}</small>"
                ).style({"font-size": "0.85em"}),
                _use,
            ],
            align="start",
        )
    return (nn_panel,)


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
def _(Image, base64, functools, io):
    # cached per (path, height): aspect-preserving JPEG data URI, for images
    # shown inside widgets (which can't use marimo's file URLs)
    @functools.cache
    def image_data_uri(path, height):
        with Image.open(path) as img:
            _im = img.convert("RGB")
            _im.thumbnail((height * 4, height), Image.LANCZOS)
        _buf = io.BytesIO()
        _im.save(_buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(_buf.getvalue()).decode()

    return (image_data_uri,)


@app.cell
def _(pl):
    def filter_images(df, doc_type=None, tags=(), description="", shelfmark=""):
        """Filter images by document type, any of the given PGP tags, and
        case-insensitive substring search on description and shelfmark"""
        if doc_type:
            df = df.filter(pl.col.doc_types.list.contains(doc_type))
        if tags:
            df = df.filter(pl.col.tags.list.set_intersection(list(tags)).list.len() > 0)
        if description.strip():
            df = df.filter(
                pl.col.description.str.to_lowercase().str.contains(
                    description.strip().lower(), literal=True
                )
            )
        if shelfmark.strip():
            df = df.filter(
                pl.col.shelfmark.str.to_lowercase().str.contains(
                    shelfmark.strip().lower(), literal=True
                )
            )
        return df

    return (filter_images,)


@app.cell
def _(IIIFImageClient, Path, data_dir, has_iiif, mo):
    # load images from the IIIF image server instead of local files, when
    # the metadata has IIIF urls (disabled otherwise); on by default when the
    # local image files aren't present
    use_iiif = mo.ui.checkbox(
        value=has_iiif and not (data_dir / "data" / "pgp-images").is_dir(),
        label="Use IIIF images",
        disabled=not has_iiif,
    )

    def iiif_url(base, width=None, height=None, fit=False):
        """IIIF Image API url for a scaled full image, from the image service
        base url; with ``fit``, scale to fit within width x height (preserving
        aspect ratio)"""
        _endpoint, _image_id = base.rstrip("/").rsplit("/", 1)
        _img = IIIFImageClient(api_endpoint=_endpoint, image_id=_image_id)
        return str(_img.size(width=width, height=height, exact=fit))

    def prefer_iiif(path, iiif):
        """whether to load an image via IIIF: when it has a IIIF url and either
        IIIF is enabled or the local file is missing"""
        return bool(iiif) and (use_iiif.value or not Path(path).exists())

    def image_src(row, height):
        """IIIF url when enabled (or the local file is missing) and available
        for this image, otherwise the local image path"""
        if prefer_iiif(row["image_path"], row.get("iiif_image")):
            return iiif_url(row["iiif_image"], height=height)
        return row["image_path"]

    return iiif_url, image_src, prefer_iiif, use_iiif


@app.function
def image_label(meta):
    """hover text for an image: whatever document metadata is available"""
    _lines = []
    if meta.get("shelfmark"):
        _lines.append(meta["shelfmark"])
    if meta.get("pgpids"):
        _lines.append("PGPID " + ", ".join(str(_p) for _p in meta["pgpids"]))
    if meta.get("doc_type"):
        _lines.append(meta["doc_type"])
    return "\n".join(_lines)


@app.cell
def _(pgp_meta_df, pl):
    def document_links_md(meta, max_chars=240):
        """Markdown for an image's shelfmark and its PGP documents: link to each
        document page, type, and the start of its description"""
        _docs = (
            pgp_meta_df.filter(pl.col.pgpid.is_in(meta.get("pgpids") or []))
            .unique("pgpid")
            .sort("pgpid")
        )
        _items = []
        for _d in _docs.iter_rows(named=True):
            _item = f"- [PGPID {_d['pgpid']}]({_d['url']})"
            if _d["type"]:
                _item += f" · {_d['type']}"
            if _d["description"]:
                _desc = _d["description"]
                _item += (
                    f"  \n  {_desc[:max_chars]}{'…' if len(_desc) > max_chars else ''}"
                )
            _items.append(_item)
        _head = f"**{meta.get('shelfmark') or 'No shelfmark match'}**"
        return (
            _head
            + "\n\n"
            + ("\n".join(_items) if _items else "*No linked PGP documents.*")
        )

    return (document_links_md,)


@app.cell
def _(ObservablePlot, Path, mo, traitlets, urllib):
    # JS shared with ../umap_to_grid.py: builds its own controls and redraws in
    # the browser, so animation never round-trips to Python
    # (fetched from github when the notebook is run on its own, e.g. on molab)
    _js_path = (mo.notebook_dir() or Path.cwd()).parent / "umap_to_grid.js"
    if _js_path.exists():
        MORPH_JS = _js_path.read_text()
    else:
        with urllib.request.urlopen(
            "https://raw.githubusercontent.com/rlskoeser/iconoscope/"
            "feature/pgp-notebook/notebooks/umap_to_grid.js"
        ) as _resp:
            MORPH_JS = _resp.read().decode()

    class SelectablePlot(ObservablePlot):
        """ObservablePlot with a synced `selected` point id; MORPH_JS makes
        points clickable when it's present (-1 = nothing selected)"""

        selected = traitlets.Int(-1).tag(sync=True)

    return MORPH_JS, SelectablePlot


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    # checklist widget adapted from ../nearest_neighbors.py
    class ModelPicker(anywidget.AnyWidget):
        """Checklist of model configs; check to show, drag (or use arrows) to reorder."""

        _esm = """
        function render({ model, el }) {
          const list = document.createElement("ul");
          list.className = "mp-list";
          el.appendChild(list);
          let dragFrom = null;

          function commit(order, selected) {
            model.set("order", order);
            model.set("selected", selected);
            model.save_changes();
          }
          function move(i, j) {
            const order = [...model.get("order")];
            if (j < 0 || j >= order.length) return;
            const [item] = order.splice(i, 1);
            order.splice(j, 0, item);
            commit(order, model.get("selected"));
          }
          function draw() {
            const items = model.get("items");
            const order = model.get("order");
            const selected = new Set(model.get("selected"));
            list.replaceChildren();
            order.forEach((label, i) => {
              const info = items[label] || {};
              const li = document.createElement("li");
              li.className = "mp-row" + (selected.has(label) ? "" : " mp-off");
              li.draggable = true;
              li.innerHTML = `
                <span class="mp-handle" title="drag to reorder">⋮⋮</span>
                <input type="checkbox" ${selected.has(label) ? "checked" : ""}>
                <span class="mp-model">${info.model ?? label}</span>
                <span class="mp-opts">${(info.options || []).map(o => `<code>${o}</code>`).join(" ") || '<span class="mp-default">defaults</span>'}</span>
                <span class="mp-meta">${info.meta ?? ""}</span>
                <span class="mp-arrows">
                  <button class="mp-up" ${i === 0 ? "disabled" : ""} title="move up">▲</button>
                  <button class="mp-down" ${i === order.length - 1 ? "disabled" : ""} title="move down">▼</button>
                </span>`;
              li.querySelector("input").addEventListener("change", (e) => {
                const sel = order.filter(l => l === label ? e.target.checked : selected.has(l));
                commit(order, sel);
              });
              li.querySelector(".mp-up").addEventListener("click", () => move(i, i - 1));
              li.querySelector(".mp-down").addEventListener("click", () => move(i, i + 1));
              li.addEventListener("dragstart", (e) => { dragFrom = i; li.classList.add("mp-dragging"); e.dataTransfer.effectAllowed = "move"; });
              li.addEventListener("dragend", () => { dragFrom = null; li.classList.remove("mp-dragging"); });
              li.addEventListener("dragover", (e) => { e.preventDefault(); li.classList.add("mp-over"); });
              li.addEventListener("dragleave", () => li.classList.remove("mp-over"));
              li.addEventListener("drop", (e) => {
                e.preventDefault(); li.classList.remove("mp-over");
                if (dragFrom !== null && dragFrom !== i) move(dragFrom, i);
              });
              list.appendChild(li);
            });
          }
          model.on("change:order", draw);
          model.on("change:selected", draw);
          model.on("change:items", draw);
          draw();
        }
        export default { render };
        """
        _css = """
        .mp-list { list-style: none; margin: 0; padding: 0; max-width: 720px; font-size: 0.9rem; }
        .mp-row { display: grid; grid-template-columns: 1.2em 1.5em 6em 1fr auto auto;
          align-items: center; gap: 0.5em; padding: 0.3em 0.5em; margin: 2px 0;
          border: 1px solid #ddd; border-radius: 6px; background: var(--background, white); }
        .mp-off { opacity: 0.5; }
        .mp-handle { cursor: grab; color: #999; user-select: none; }
        .mp-model { font-weight: 600; }
        .mp-opts code { font-size: 0.8em; padding: 0 0.3em; border-radius: 3px;
          color: inherit; background: color-mix(in srgb, #1f77b4 20%, transparent);
          border: 1px solid color-mix(in srgb, #1f77b4 45%, transparent); }
        .mp-default { color: #888; font-style: italic; font-size: 0.85em; }
        .mp-meta { color: #777; font-size: 0.8em; white-space: nowrap; }
        .mp-arrows button { border: none; background: none; cursor: pointer; color: #666; padding: 0 2px; }
        .mp-arrows button:disabled { color: #ddd; cursor: default; }
        .mp-dragging { opacity: 0.3; }
        .mp-over { border-color: #1f77b4; box-shadow: 0 -2px 0 #1f77b4; }
        """
        items = traitlets.Dict().tag(sync=True)
        order = traitlets.List(traitlets.Unicode()).tag(sync=True)
        selected = traitlets.List(traitlets.Unicode()).tag(sync=True)

    return (ModelPicker,)


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    # results grid widget: one row per model config, click a thumbnail to
    # select it (click again to clear); hovering highlights every copy of an image
    class NeighborGrid(anywidget.AnyWidget):
        """Rows of neighbor thumbnails; syncs the clicked image index back to Python."""

        _esm = """
        function render({ model, el }) {
          const grid = document.createElement("div");
          grid.className = "nn-grid";
          el.appendChild(grid);
          function draw() {
            const rows = model.get("rows");
            const sel = model.get("selected");
            const h = model.get("thumb_height");
            const showCaps = model.get("show_captions");
            grid.replaceChildren();
            for (const row of rows) {
              const label = document.createElement("div");
              label.className = "nn-label";
              const m = document.createElement("div");
              m.className = "nn-model";
              m.textContent = row.model;
              label.appendChild(m);
              for (const o of row.options) {
                const d = document.createElement("div");
                d.className = "nn-opt";
                d.textContent = o;
                label.appendChild(d);
              }
              const imgs = document.createElement("div");
              imgs.className = "nn-imgs";
              for (const t of row.thumbs) {
                const th = document.createElement("div");
                th.className = "nn-thumb" + (t.idx === sel ? " nn-selected" : "");
                th.dataset.idx = t.idx;
                th.title = t.title;
                const img = document.createElement("img");
                img.src = t.src;
                img.style.height = h + "px";
                th.appendChild(img);
                if (showCaps) {
                  const c = document.createElement("span");
                  c.className = "nn-cap";
                  c.textContent = `#${t.rank} · ${t.similarity.toFixed(3)}`;
                  th.appendChild(c);
                }
                if (t.count > 1) {
                  const b = document.createElement("span");
                  b.className = "nn-count";
                  b.textContent = `×${t.count}`;
                  th.appendChild(b);
                }
                th.addEventListener("mouseenter", () => {
                  grid.classList.add("nn-hovering");
                  grid.querySelectorAll(`.nn-thumb[data-idx="${t.idx}"]`)
                    .forEach((e) => e.classList.add("nn-hot"));
                });
                th.addEventListener("mouseleave", () => {
                  grid.classList.remove("nn-hovering");
                  grid.querySelectorAll(".nn-hot").forEach((e) => e.classList.remove("nn-hot"));
                });
                th.addEventListener("click", () => {
                  model.set("selected", t.idx === model.get("selected") ? -1 : t.idx);
                  model.save_changes();
                });
                imgs.appendChild(th);
              }
              grid.append(label, imgs);
            }
          }
          for (const k of ["rows", "selected", "thumb_height", "show_captions"])
            model.on(`change:${k}`, draw);
          draw();
        }
        export default { render };
        """
        _css = """
        .nn-grid { display: grid; grid-template-columns: max-content 1fr;
          column-gap: 0.5rem; row-gap: 4px; align-items: center; }
        .nn-label { text-align: right; font-size: 0.8rem; line-height: 1.2; }
        .nn-model { font-weight: 600; }
        .nn-opt { font-size: 0.7rem; opacity: 0.7; }
        .nn-imgs { display: flex; gap: 2px; overflow-x: auto; }
        .nn-thumb { position: relative; flex: none; cursor: pointer;
          transition: opacity 0.1s; }
        .nn-thumb img { margin: 0; display: block; width: auto; }
        .nn-cap { position: absolute; left: 0; bottom: 0; padding: 0 3px;
          font-size: 0.6rem; color: white; background: rgba(0, 0, 0, 0.55); }
        .nn-count { position: absolute; right: 2px; top: 2px; padding: 0 4px;
          border-radius: 8px; font-size: 0.65rem; font-weight: 600;
          color: white; background: #ff7f0e; }
        .nn-hovering .nn-thumb { opacity: 0.35; }
        .nn-hovering .nn-thumb.nn-hot { opacity: 1; }
        .nn-hovering .nn-thumb.nn-hot.nn-shared,
        .nn-thumb.nn-hot:has(.nn-count) { outline: 3px solid #ff7f0e; outline-offset: -3px; }
        .nn-thumb.nn-selected { outline: 3px solid #1f77b4; outline-offset: -3px; opacity: 1; }
        """
        rows = traitlets.List().tag(sync=True)
        selected = traitlets.Int(-1).tag(sync=True)
        thumb_height = traitlets.Int(120).tag(sync=True)
        show_captions = traitlets.Bool(True).tag(sync=True)

    return (NeighborGrid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    Set one of these environment variables before starting the notebook:

    - **`PGP_DATA_URL`** — url (or local path) of a data tarball, e.g. the
      sample subset made with `make_molab_subset.py`; downloaded and extracted to
      `~/.cache/iconoscope/pgp-subset` on first use. To switch to a different
      tarball, delete that directory.
    - **`PGP_DATA_DIR`** — local directory with images under `data/pgp-images/`
      and the dataset + image metadata files: `pgp.h5` and `image_meta.csv`, or
      override the filenames with **`PGP_DATASET`** (e.g. `pgp_v2.h5`) and
      **`PGP_IMAGE_META`** (e.g. `image_meta_v2.parquet`, which includes IIIF
      image urls)

    When the image metadata includes IIIF image urls, check *Use IIIF images* to
    load images from the IIIF image server instead of local files.

    For example:

    ```sh
    PGP_DATA_URL=https://example.com/pgp-molab-subset.tar \
      uvx marimo edit --sandbox pgp.py
    ```

    PGP document metadata is loaded from the
    [pgp-metadata](https://github.com/princetongenizalab/pgp-metadata) repository.
    """)
    return


@app.cell
def _():
    import base64
    import functools
    import io
    import re
    import urllib.request
    from pathlib import Path

    import anywidget
    import marimo as mo
    import numpy as np
    import polars as pl
    import traitlets
    from piffle.image import IIIFImageClient
    from PIL import Image, ImageOps
    from wigglystuff import ObservablePlot

    from iconoscope.dataset import ImageDataset
    from iconoscope.mosaic import layout_mosaic, render_mosaic

    # mosaic canvas size in pixels; 16:10, close to typical screens, so the plot
    # fills most of the screen in fullscreen
    CANVAS_WIDTH = 2400
    CANVAS_HEIGHT = 1500
    # max on-screen size of the plot widget, in CSS pixels
    PLOT_MAX_WIDTH = 1400
    PLOT_HEIGHT = 900
    # downloadable mosaic of the full filtered set: target cell width in px,
    # with the canvas (16:10, like the plot) capped at MOSAIC_MAX_WIDTH
    # (memory: ~3 bytes/px, so 16000 x 10000 is ~480 MB while rendering)
    MOSAIC_CELL_WIDTH = 250
    MOSAIC_MAX_WIDTH = 16000
    # thumbnail resolution (max width in px); high enough to stay sharp when
    # zoomed into the plot, but each image is embedded in the page
    THUMB_PX = 384
    # space above the plot for the summary line and controls
    PLOT_CHROME = 64
    return (
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        IIIFImageClient,
        Image,
        ImageDataset,
        ImageOps,
        MOSAIC_CELL_WIDTH,
        MOSAIC_MAX_WIDTH,
        ObservablePlot,
        PLOT_CHROME,
        PLOT_HEIGHT,
        PLOT_MAX_WIDTH,
        Path,
        THUMB_PX,
        anywidget,
        base64,
        functools,
        io,
        layout_mosaic,
        mo,
        np,
        pl,
        re,
        render_mosaic,
        traitlets,
        urllib,
    )


@app.cell
def _(Path, urllib):
    import os
    import tarfile

    # data location from environment variables; see "Data" section above
    # PGP document metadata is always loaded from the pgp-metadata repo
    if os.environ.get("PGP_DATA_DIR"):
        data_dir = Path(os.environ["PGP_DATA_DIR"]).expanduser()
        # optional dataset / metadata filenames within the data dir
        dataset_name = os.environ.get("PGP_DATASET", "pgp.h5")
        image_meta_name = os.environ.get("PGP_IMAGE_META", "image_meta.csv")
    elif os.environ.get("PGP_DATA_URL"):
        _url = os.environ["PGP_DATA_URL"]
        data_dir = Path.home() / ".cache" / "iconoscope" / "pgp-subset"
        dataset_name = "pgp.h5"
        image_meta_name = "image_meta.csv"
        if not (data_dir / "pgp.h5").exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            _tar_path = data_dir / "download.tar"
            if Path(_url).expanduser().exists():
                _tar_path = Path(_url).expanduser()
            else:
                urllib.request.urlretrieve(_url, _tar_path)
            with tarfile.open(_tar_path) as _tar:
                _tar.extractall(data_dir, filter="data")
            if _tar_path.parent == data_dir:
                _tar_path.unlink()
    else:
        raise RuntimeError(
            "No data configured: set PGP_DATA_DIR (local data directory) or "
            "PGP_DATA_URL (data tarball url or path) and restart the notebook"
        )
    documents_csv = (
        "https://github.com/princetongenizalab/pgp-metadata/raw/main/data/documents.csv"
    )
    image_meta_path = data_dir / image_meta_name
    return data_dir, dataset_name, documents_csv, image_meta_path


@app.cell
def _(ImageDataset, data_dir, dataset_name):
    import h5py

    pgp_dataset_path = data_dir / dataset_name
    # image paths are stored relative to image_dir; point it at this copy
    # of the images (no-op for the original dataset)
    _image_dir = str(data_dir / "data" / "pgp-images")
    with h5py.File(pgp_dataset_path, "r") as _f:
        _needs_update = _f["image"].attrs["image_dir"] != _image_dir
    if _needs_update:
        with h5py.File(pgp_dataset_path, "r+") as _f:
            _f["image"].attrs["image_dir"] = _image_dir
    img_dataset = ImageDataset(pgp_dataset_path)
    return (img_dataset,)


@app.cell
def _(image_meta_path, pl):
    # load image metadata (csv or parquet)
    if image_meta_path.suffix == ".parquet":
        # v2 metadata: clean columns, including IIIF image urls
        image_meta_df = pl.read_parquet(image_meta_path)
    else:
        # TODO: this csv changes structure part way through and is not all of the files
        image_meta_df = pl.read_csv(image_meta_path)
        # fix swapped image_path / shelfmark columns
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
    # IIIF image service url, when available (null otherwise)
    if "iiif_image" not in image_meta_df.columns:
        image_meta_df = image_meta_df.with_columns(iiif_image=pl.lit(None, pl.String))
    has_iiif = image_meta_df["iiif_image"].is_not_null().any()
    return has_iiif, image_meta_df


@app.cell
def _(documents_csv, pl):
    pgp_meta_df = pl.read_csv(documents_csv, infer_schema_length=None)
    # split and explode on shelfmark so we get one row per document+shelfmark combination
    pgp_meta_df = (
        pgp_meta_df.with_columns(shelfmark_list=pl.col.shelfmark.str.split(" + "))
        .explode("shelfmark_list")
        .rename({"shelfmark": "shelfmark_orig", "shelfmark_list": "shelfmark"})
    )
    return (pgp_meta_df,)


@app.cell
def _(image_meta_df, pgp_meta_df, pl):
    # one row per image (image_meta lists some images more than once), with
    # PGP document info attached by shelfmark
    _img = image_meta_df.select("image_path", "shelfmark", "iiif_image").unique(
        "image_path", keep="first"
    )
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
            # descriptions of all documents on the shelfmark, for text search
            pl.col.description.drop_nulls().str.join("\n").alias("description"),
        )
        .with_columns(doc_type=pl.col.doc_types.list.join(", "))
    )
    image_info_df = _img.join(_docs, on="shelfmark", how="left")
    return (image_info_df,)


if __name__ == "__main__":
    app.run()
