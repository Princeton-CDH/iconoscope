# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "iconoscope",
#     "numpy==2.5.4",
#     "polars==2.0.0",
# ]
#
# [tool.uv.sources]
# iconoscope = { path = "..", editable = true }
# ///
"""Compare nearest neighbors for a selected image across all embedding models
present in an iconoscope dataset.

Run from the repo root with:
    uv run --with marimo --with altair marimo edit notebooks/nearest_neighbors.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import random
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    from iconoscope.dataset import ImageDataset
    from iconoscope.models import AVAILABLE_MODELS

    # background points are sampled/capped in the plotting cell
    alt.data_transformers.disable_max_rows()

    return AVAILABLE_MODELS, ImageDataset, Path, alt, mo, np, pl, random


@app.cell
def _(mo):
    mo.md("""
    # Nearest neighbors across embeddings

    Open an iconoscope dataset (`.h5`), choose an image, and compare its
    nearest neighbors (cosine similarity) for every embedding model
    stored in the dataset.
    """)
    return


@app.cell
def _(mo):
    dataset_path = mo.ui.text(
        label="Dataset path (.h5)", placeholder="path/to/dataset.h5", full_width=True
    )
    dataset_path
    return (dataset_path,)


@app.cell
def _(AVAILABLE_MODELS, ImageDataset, Path, dataset_path, mo):
    _path = Path(dataset_path.value.strip()).expanduser()
    mo.stop(
        not dataset_path.value.strip() or not _path.is_file(),
        mo.callout(mo.md("Enter the path to an existing dataset file."), kind="warn"),
    )

    dataset = ImageDataset(storage_path=_path)
    info = dataset.info()
    # only models that have stored features
    models = [
        AVAILABLE_MODELS[name]
        for name, data in info["models"].items()
        if "embeddings" in data and name in AVAILABLE_MODELS
    ]
    mo.stop(
        not models,
        mo.callout(
            mo.md("No embedding features found in this dataset."), kind="danger"
        ),
    )
    mo.md(
        f"**{info['image_paths']:,}** images from `{info['image_dir']}`; "
        f"models: {', '.join(f'`{m.name}`' for m in models)}"
    )
    return dataset, models


@app.cell
def _(dataset, models, np, pl):
    # load paths once, then features + umap for each model
    image_df = dataset.get_image_data().with_row_index("idx")
    model_data = {}
    for _model in models:
        _df = dataset.load_data(model=_model, features=True, umap=True)
        _feats = _df["features"].to_numpy().astype(np.float32)
        # L2-normalize so dot product == cosine similarity
        _feats /= np.linalg.norm(_feats, axis=1, keepdims=True).clip(min=1e-12)
        _umap = _df["umap"].to_numpy()
        model_data[_model.name] = {
            "features": _feats,
            "umap": pl.DataFrame(
                {
                    "idx": np.arange(len(_umap)),
                    "umap1": _umap[:, 0],
                    "umap2": _umap[:, 1],
                }
            ),
        }
    return image_df, model_data


@app.cell
def _(mo):
    random_button = mo.ui.button(
        label="Random image", value=0, on_click=lambda v: v + 1
    )
    k_slider = mo.ui.slider(3, 30, value=10, label="Neighbors (k)", show_value=True)
    thumb_size = mo.ui.slider(
        60, 250, value=120, step=10, label="Thumbnail height", show_value=True
    )
    return k_slider, random_button, thumb_size


@app.cell
def _(image_df, mo, random, random_button):
    # re-created when the random button is clicked, picking a new default
    random_button.value
    _start = random.randrange(image_df.height)
    image_index = mo.ui.number(
        start=0, stop=image_df.height - 1, value=_start, label="Image index"
    )
    return (image_index,)


@app.cell
def _(image_df, image_index, k_slider, mo, random_button, thumb_size):
    query_idx = int(image_index.value)
    query_path = image_df["image_path"][query_idx]
    mo.hstack(
        [
            mo.vstack([image_index, random_button, k_slider, thumb_size]),
            mo.vstack(
                [
                    # fixed height, auto width -> preserve aspect ratio
                    mo.image(
                        query_path,
                        style={
                            "height": "250px",
                            "width": "auto",
                            "object-fit": "contain",
                        },
                    ),
                    mo.md(f"`{query_path}`").style({"font-size": "0.8em"}),
                ]
            ),
        ],
        justify="start",
        gap=2,
    )
    return (query_idx,)


@app.cell
def _(image_df, k_slider, model_data, np, pl, query_idx):
    # nearest neighbors per model (excluding the query image itself)
    neighbors = {}
    for _name, _data in model_data.items():
        _sims = _data["features"] @ _data["features"][query_idx]
        _sims[query_idx] = -np.inf
        _k = k_slider.value
        _top = np.argpartition(-_sims, _k)[:_k]
        _top = _top[np.argsort(-_sims[_top])]
        neighbors[_name] = pl.DataFrame(
            {"idx": _top, "rank": np.arange(1, _k + 1), "similarity": _sims[_top]}
        ).join(image_df.select("idx", "image_path"), on="idx", how="left")
    return (neighbors,)


@app.cell
def _(mo, neighbors, thumb_size):
    def _thumb(row):
        return mo.vstack(
            [
                mo.image(
                    row["image_path"],
                    style={
                        "height": f"{thumb_size.value}px",
                        "width": "auto",
                        "object-fit": "contain",
                    },
                ),
                mo.Html(
                    f"<span style='font-size: 0.7rem; color: #666'>"
                    f"#{row['rank']} · {row['similarity']:.3f}</span>"
                ),
            ],
            gap=0.1,
            align="center",
        )

    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(f"### {_name}"),
                    mo.hstack(
                        [_thumb(r) for r in _df.iter_rows(named=True)],
                        justify="start",
                        wrap=False,
                        gap=0.5,
                    ).style({"overflow-x": "auto"}),
                ]
            )
            for _name, _df in neighbors.items()
        ]
    )
    return


@app.cell
def _(mo):
    umap_view = mo.ui.radio(
        options=["combined", "separate"],
        value="combined",
        label="UMAP view",
        inline=True,
    )
    show_background = mo.ui.checkbox(value=True, label="Show other images")
    max_background = mo.ui.slider(
        500,
        20000,
        value=3000,
        step=500,
        label="Max background points per model",
        show_value=True,
    )
    zoom = mo.ui.slider(
        0.05,
        1.0,
        value=1.0,
        step=0.05,
        label="View radius (umap units)",
        show_value=True,
    )
    mo.hstack(
        [umap_view, show_background, max_background, zoom], justify="start", gap=2
    )
    return max_background, show_background, umap_view, zoom


@app.cell
def _(image_df, model_data, neighbors, pl, query_idx):
    # Long dataframe of all models' UMAP coords, each re-centered so that
    # the query image sits at (0, 0). UMAP coords are already normalized to
    # [0, 1] per model, so the centered offsets are on comparable scales.
    # Note: UMAP orientation is arbitrary, so only distance from the
    # center is meaningful when comparing models — not direction.
    _frames = []
    for _name, _data in model_data.items():
        _umap = _data["umap"]
        _q = _umap.filter(pl.col.idx == query_idx).row(0, named=True)
        _nbr = neighbors[_name].select("idx", "rank", "similarity")
        _frames.append(
            _umap.join(_nbr, on="idx", how="left").with_columns(
                model=pl.lit(_name),
                dx=pl.col.umap1 - _q["umap1"],
                dy=pl.col.umap2 - _q["umap2"],
                role=pl.when(pl.col.idx == query_idx)
                .then(pl.lit("query"))
                .when(pl.col.rank.is_not_null())
                .then(pl.lit("neighbor"))
                .otherwise(pl.lit("other")),
            )
        )
    umap_df = (
        pl.concat(_frames)
        .join(image_df.select("idx", "image_path"), on="idx", how="left")
        .with_columns(dist=(pl.col.dx**2 + pl.col.dy**2).sqrt())
    )
    return (umap_df,)


@app.cell
def _(alt, max_background, mo, pl, show_background, umap_df, umap_view, zoom):
    _r = zoom.value
    _scale = alt.Scale(domain=[-_r, _r])
    _x = alt.X("dx:Q", scale=_scale, title=None)
    _y = alt.Y("dy:Q", scale=_scale, title=None)
    _tooltip = [
        "model",
        "idx",
        "rank",
        alt.Tooltip("similarity:Q", format=".3f"),
        "image_path",
    ]

    # restrict to the visible window (keeps the payload small when zoomed)
    _df = umap_df.filter((pl.col.dx.abs() <= _r) & (pl.col.dy.abs() <= _r))
    # background: only the columns needed to draw, rounded, and sampled
    # per model, so chart JSON stays small for large datasets
    _bg_df = (
        _df.filter(pl.col.role == "other")
        .select("model", pl.col("dx", "dy").round(4))
        .filter(
            pl.int_range(pl.len()).shuffle(seed=0).over("model") < max_background.value
        )
    )
    if not show_background.value:
        _bg_df = _bg_df.clear()
    _fg_df = _df.filter(pl.col.role == "neighbor")
    _q_df = _df.filter(pl.col.role == "query").head(1)

    _crosshair = alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(
        color="#999", strokeDash=[4, 4]
    )
    _rules = _crosshair.encode(x="z:Q") + _crosshair.encode(y="z:Q")
    _query = (
        alt.Chart(_q_df)
        .mark_point(shape="cross", size=250, color="#d62728", filled=True)
        .encode(x=_x, y=_y, tooltip=["image_path"])
    )

    def _layers(bg, fg, color):
        layers = [_rules]
        if show_background.value and bg.height:
            layers.append(
                alt.Chart(bg)
                .mark_circle(size=6, opacity=0.25)
                .encode(x=_x, y=_y, color=color)
            )
        layers.append(
            alt.Chart(fg)
            .mark_circle(size=90, opacity=0.9, stroke="black", strokeWidth=0.5)
            .encode(x=_x, y=_y, color=color, tooltip=_tooltip)
        )
        layers.append(_query)
        return alt.layer(*layers).resolve_scale(color="shared")

    if umap_view.value == "combined":
        # one plot: every model's neighbors overlaid, colored by model
        _color = alt.Color("model:N", legend=alt.Legend(title="model"))
        _chart = (
            _layers(_bg_df, _fg_df, _color)
            .properties(
                width=600,
                height=600,
                title="Neighbors per model (query centered at 0,0)",
            )
            .interactive()
        )
        _out = _chart
    else:
        # small multiples, each centered on the query with a shared scale
        _charts = []
        for _name in umap_df["model"].unique(maintain_order=True):
            # background in grey, neighbors in blue
            _c = alt.layer(
                _rules,
                *(
                    [
                        alt.Chart(_bg_df.filter(pl.col.model == _name))
                        .mark_circle(size=6, opacity=0.3, color="#bbbbbb")
                        .encode(x=_x, y=_y)
                    ]
                    if show_background.value
                    else []
                ),
                alt.Chart(_fg_df.filter(pl.col.model == _name))
                .mark_circle(size=80, color="#1f77b4", stroke="black", strokeWidth=0.5)
                .encode(x=_x, y=_y, tooltip=_tooltip),
                _query,
            ).properties(title=_name, width=320, height=320)
            _charts.append(_c)
        _out = mo.hstack(_charts, wrap=True, justify="start")

    _out
    return


@app.cell
def _(alt, pl, umap_df):
    # Distance from query in UMAP space vs. neighbor rank, per model —
    # an orientation-independent way to compare models on one plot.
    _df = umap_df.filter(pl.col.role == "neighbor")
    alt.Chart(_df).mark_line(point=True).encode(
        x=alt.X("rank:O", title="neighbor rank"),
        y=alt.Y("dist:Q", title="UMAP distance from query"),
        color="model:N",
        tooltip=[
            "model",
            "rank",
            "image_path",
            alt.Tooltip("similarity:Q", format=".3f"),
        ],
    ).properties(
        width=600, height=200, title="How tightly do neighbors cluster in UMAP?"
    )
    return


if __name__ == "__main__":
    app.run()
