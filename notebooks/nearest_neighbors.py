# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "iconoscope",
#     "numpy==2.5.4",
#     "polars==2.0.0",
#     "anywidget==0.11.0",
#     "traitlets==5.16.1",
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
    import anywidget
    import traitlets
    import marimo as mo
    import numpy as np
    import polars as pl

    from iconoscope.dataset import ImageDataset

    # background points are sampled/capped in the plotting cell
    alt.data_transformers.disable_max_rows()
    return ImageDataset, Path, alt, anywidget, mo, np, pl, random, traitlets


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
def _(ImageDataset, Path, dataset_path, mo):
    _path = Path(dataset_path.value.strip()).expanduser()
    mo.stop(
        not dataset_path.value.strip() or not _path.is_file(),
        mo.callout(mo.md("Enter the path to an existing dataset file."), kind="warn"),
    )

    dataset = ImageDataset(storage_path=_path)
    info = dataset.info()
    # model + preprocessing configs with stored features, keyed by label;
    # sorted by model name, with default preprocessing first
    models = {
        f"{config.model.name}/{config.key}": config
        for config in sorted(
            dataset.get_configs(),
            key=lambda c: (c.model.name, c.key != "defaults", c.key),
        )
    }
    mo.stop(
        not models,
        mo.callout(
            mo.md("No embedding features found in this dataset."), kind="danger"
        ),
    )
    mo.md(
        f"**{info['image_paths']:,}** images from `{info['image_dir']}`; "
        f"**{len(models)}** model configurations with embeddings"
    )
    return dataset, info, models


@app.cell(hide_code=True)
def _(anywidget, traitlets):
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
def _(ModelPicker, info, mo, models):
    def _model_item(label, config):
        _data = info["models"].get(label, {})
        _meta = [f"{config.model.dimensions}d"]
        if "umap" in _data:
            _meta.append("umap")
        if "cluster" in _data:
            _meta.append(f"k={_data['cluster']['k']}")
        return {
            "model": config.model.name,
            # one badge per set option (key can't be split on "_": center_crop)
            "options": [
                _opt
                if _val is True
                else f"{_opt}={_val:g}"
                if isinstance(_val, float)
                else f"{_opt}={_val}"
                for _opt, _val in config.opts.items()
                if _val is not None and _val is not False
            ],
            "meta": " · ".join(_meta),
        }

    model_picker = mo.ui.anywidget(
        ModelPicker(
            items={_l: _model_item(_l, _c) for _l, _c in models.items()},
            order=list(models),
            selected=list(models),
        )
    )
    mo.vstack(
        [
            mo.md("**Models** — check to display; drag or use arrows to reorder"),
            model_picker,
        ]
    )
    return (model_picker,)


@app.cell(hide_code=True)
def _(mo, model_picker):
    # checked model configs, in display order
    selected_models = [
        _l for _l in model_picker.value["order"] if _l in model_picker.value["selected"]
    ]
    mo.stop(
        not selected_models,
        mo.callout(mo.md("Select at least one model to display."), kind="warn"),
    )
    return (selected_models,)


@app.cell
def _(dataset, models, np, pl):
    # load paths once, then features + umap for each model
    image_df = dataset.get_image_data().with_row_index("idx")
    model_data = {}
    for _label, _config in models.items():
        _df = dataset.load_data(config=_config, features=True, umap=True)
        _feats = _df["features"].to_numpy().astype(np.float32)
        # L2-normalize so dot product == cosine similarity
        _feats /= np.linalg.norm(_feats, axis=1, keepdims=True).clip(min=1e-12)
        _umap = _df["umap"].to_numpy()
        model_data[_label] = {
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
def _(image_df, k_slider, model_data, np, pl, query_idx, selected_models):
    # nearest neighbors per model (excluding the query image itself)
    neighbors = {}
    for _name in selected_models:
        _data = model_data[_name]
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
def _(image_df, model_data, neighbors, pl, query_idx, selected_models):
    # Long dataframe of all models' UMAP coords, each re-centered so that
    # the query image sits at (0, 0). UMAP coords are already normalized to
    # [0, 1] per model, so the centered offsets are on comparable scales.
    # Note: UMAP orientation is arbitrary, so only distance from the
    # center is meaningful when comparing models — not direction.
    _frames = []
    for _name in selected_models:
        _umap = model_data[_name]["umap"]
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
def _(
    alt,
    max_background,
    mo,
    pl,
    selected_models,
    show_background,
    umap_df,
    umap_view,
    zoom,
):
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
        _color = alt.Color(
            "model:N",
            sort=selected_models,
            scale=alt.Scale(domain=selected_models),
            legend=alt.Legend(title="model"),
        )
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
        for _name in selected_models:
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
def _(alt, pl, selected_models, umap_df):
    # Distance from query in UMAP space vs. neighbor rank, per model —
    # an orientation-independent way to compare models on one plot.
    _df = umap_df.filter(pl.col.role == "neighbor")
    alt.Chart(_df).mark_line(point=True).encode(
        x=alt.X("rank:O", title="neighbor rank"),
        y=alt.Y("dist:Q", title="UMAP distance from query"),
        color=alt.Color("model:N", scale=alt.Scale(domain=selected_models)),
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
