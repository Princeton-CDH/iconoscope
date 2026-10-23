# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "wigglystuff",
#     "iconoscope",
#     "numpy==2.5.4",
#     "pillow==12.3.0",
# ]
#
# [tool.uv.sources]
# iconoscope = { path = "..", editable = true }
# ///
"""Visualize how UMAP coordinates are slotted into the mosaic grid.

Run from the repo root with:
    uv run --with marimo --with wigglystuff marimo edit notebooks/umap_to_grid.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import io
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from PIL import Image, ImageOps
    from wigglystuff import ObservablePlot

    from iconoscope.dataset import ImageDataset
    from iconoscope.models import DEFAULT_MODEL
    from iconoscope.mosaic import layout_mosaic

    # max on-screen size of the plot widget, in CSS pixels
    PLOT_MAX_WIDTH = 1400
    PLOT_HEIGHT = 700
    # space above the plot for the summary line and controls
    PLOT_CHROME = 64
    return (
        DEFAULT_MODEL,
        Image,
        ImageDataset,
        ImageOps,
        ObservablePlot,
        PLOT_CHROME,
        PLOT_HEIGHT,
        PLOT_MAX_WIDTH,
        Path,
        base64,
        io,
        layout_mosaic,
        mo,
        np,
    )


@app.cell
def _(mo):
    mo.md("""
    # UMAP → grid

    Each image has a 2D UMAP coordinate (normalized 0–1). `assign_to_grid` solves a
    linear assignment (lapjv) to move every point to a unique grid cell while
    minimizing total distance. Drag **t** (or press ▶) to morph from UMAP
    positions (t=0) to grid slots (t=1). Colors encode the *original* UMAP
    position, so a smooth gradient in the grid means neighborhoods were preserved.
    """)
    return


@app.cell
def _(mo):
    dataset_path = mo.ui.text(
        label="Dataset (.h5) — leave blank for synthetic demo", full_width=True
    )
    n_sample = mo.ui.slider(
        20, 1000, value=150, step=10, label="Sample size", show_value=True
    )
    canvas_w = mo.ui.number(200, 8000, value=2500, label="Canvas width")
    canvas_h = mo.ui.number(200, 8000, value=1000, label="Canvas height")
    show_thumbs = mo.ui.checkbox(value=False, label="Thumbnails (real data)")
    mo.vstack([dataset_path, mo.hstack([n_sample, canvas_w, canvas_h, show_thumbs])])
    return canvas_h, canvas_w, dataset_path, n_sample, show_thumbs


@app.cell
def _(DEFAULT_MODEL, ImageDataset, Path, dataset_path, mo):
    # embedding configs (model + preprocessing) stored in the dataset
    _path = Path(dataset_path.value.strip()).expanduser()
    dataset = None
    embedding = None
    if dataset_path.value.strip():
        mo.stop(not _path.is_file(), mo.md(f"**Dataset not found:** `{_path}`"))
        dataset = ImageDataset(storage_path=_path)
        # only configs with stored umap can be plotted
        _configs = dataset.configs_by_label(require_umap=True)
        mo.stop(
            not _configs,
            mo.md(
                "**No UMAP found in this dataset;** "
                "run `iconoscope mosaic` to generate."
            ),
        )
        # default model with default preprocessing if present, else first
        _default = next(
            (
                _l
                for _l, _c in _configs.items()
                if _c.model == DEFAULT_MODEL and _c.key == "defaults"
            ),
            next(iter(_configs)),
        )
        embedding = mo.ui.dropdown(options=_configs, value=_default, label="Embedding")
    embedding
    return dataset, embedding


@app.cell
def _(dataset, embedding, mo, n_sample, np):
    if dataset is not None:
        with mo.status.spinner(f"Loading {embedding.selected_key}"):
            _df = dataset.load_data(embedding.value, umap=True)
        # sample after loading, so umap coords are generated for all images
        # (as in generate_mosaic)
        _df = _df.sample(min(n_sample.value, _df.height))
        coords = _df["umap"].to_numpy().astype(np.float32)
        paths = _df["image_path"].to_list()
        # same rule as generate_mosaic: most common aspect ratio in the data
        img_aspect = float(_df["aspect_ratio"].mode()[0])
    else:
        # synthetic: a few gaussian blobs, normalized 0-1 like reduce_features
        _rng = np.random.default_rng(0)
        _n = n_sample.value
        _centers = _rng.uniform(0.15, 0.85, size=(5, 2))
        _pts = _centers[_rng.integers(0, 5, _n)] + _rng.normal(0, 0.07, (_n, 2))
        _pts = (_pts - _pts.min(0)) / np.ptp(_pts, axis=0)
        coords = _pts.astype(np.float32)
        paths = None
        img_aspect = 1.0
    mo.md(f"**{len(coords)}** points, image aspect ratio {img_aspect:.2f}")
    return coords, img_aspect, paths


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
def _(
    PLOT_CHROME,
    PLOT_HEIGHT,
    PLOT_MAX_WIDTH,
    Image,
    ImageOps,
    base64,
    cell_h,
    cell_w,
    io,
    layout,
    mo,
    paths,
    show_thumbs,
):
    # base64 thumbnails sized to the cell as drawn on screen at 2x for high-DPI
    # displays (never larger than the mosaic cell). The full image is kept,
    # scaled to *cover* the cell without cropping, so the browser can either
    # crop it to fill the cell (like the mosaic) or fit it whole; never distorted.
    def load_uncropped(path, size):
        with Image.open(path) as img:
            return ImageOps.cover(img.convert("RGB"), size, Image.LANCZOS)

    thumbs = None
    if paths is not None and show_thumbs.value:
        _k = min(
            PLOT_MAX_WIDTH / layout.canvas_width,
            (PLOT_HEIGHT - PLOT_CHROME) / layout.canvas_height,
        )
        _f = min(1.0, 2 * _k)
        _size = (max(1, round(cell_w * _f)), max(1, round(cell_h * _f)))
        thumbs = []
        _errors = []
        for _p in paths:
            try:
                _im = load_uncropped(_p, _size)
                _buf = io.BytesIO()
                _im.save(_buf, format="JPEG", quality=85)
                thumbs.append(
                    "data:image/jpeg;base64,"
                    + base64.b64encode(_buf.getvalue()).decode()
                )
            except Exception as _exc:
                thumbs.append(None)
                _errors.append(f"{_p}: {_exc}")
    if thumbs is not None:
        _ok = sum(t is not None for t in thumbs)
        _msg = f"Loaded **{_ok}/{len(thumbs)}** thumbnails."
        if _errors:
            _msg += f" First error: `{_errors[0]}`"
        _out = mo.md(_msg)
    else:
        _out = None
    _out
    return (thumbs,)


@app.cell
def _(coords, dest, np, paths, placed, scale, thumbs):
    # one record per placed image: start (sx, sy), end (dx, dy), color, optional img
    _idx = np.flatnonzero(placed)
    _src = coords[_idx] * scale
    _dst = dest[_idx]
    _c = coords[_idx]
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
            "label": paths[i] if paths else f"item {i}",
        }
        for k, i in enumerate(_idx)
    ]
    return (points,)


@app.cell
def _(mo):
    # JS expression shared with the web export: builds its own controls and
    # redraws in the browser, so animation never round-trips to Python.
    MORPH_JS = (mo.notebook_dir() / "umap_to_grid.js").read_text()
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
    # (PLOT_CHROME px reserved for summary + controls)
    _k = min(
        PLOT_MAX_WIDTH / layout.canvas_width,
        (PLOT_HEIGHT - PLOT_CHROME) / layout.canvas_height,
    )
    widget = ObservablePlot(
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
    # summary line is drawn by the widget (same as in the web export)
    mo.ui.anywidget(widget)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Export for the web

    Each export saves the current grid under a name, as
    `umap_grid_<name>.json` (layout + inline thumbnails). Exporting again with
    the same name replaces it. After every export, these are regenerated to
    include **all** grids in the folder, with a dropdown to switch between them:

    - `umap_grid_observable.js` — one cell for an Observable notebook; attach
      every `umap_grid_*.json` file (same names), then paste the cell
    - `index.html` — standalone page that loads the JSON files next to it
      (host the folder, or preview with `python -m http.server` in the folder)

    Labels are exported as file names only, so local paths aren't published.
    """)
    return


@app.cell
def _(Path, dataset_path, embedding, mo):
    # default name: dataset + model config, e.g. "mydata_clip_grayscale"
    _stem = (
        f"{Path(dataset_path.value.strip()).stem}_"
        f"{embedding.value.model.name}_{embedding.value.key}"
        if embedding is not None
        else "synthetic"
    )
    export_dir = mo.ui.text(
        value=str(mo.notebook_dir() / "export"), label="Export folder", full_width=True
    )
    export_name = mo.ui.text(
        value=_stem, label="Name (used in file name: letters, digits, - _)"
    )
    export_title = mo.ui.text(value=_stem, label="Title (shown in dropdown)")
    export_button = mo.ui.run_button(label="Export")
    mo.vstack([export_dir, mo.hstack([export_name, export_title]), export_button])
    return export_button, export_dir, export_name, export_title


@app.cell
def _(MORPH_JS, PLOT_CHROME):
    import json

    def render_fn_js() -> str:
        """JS function that renders one grid's data with the morph animation."""
        # MORPH_JS expects its inputs as variables in scope; provide them as
        # parameters. Open paren on the return line: the JS starts with comments.
        return (
            "function renderGrid(data, width, invalidation) {\n"
            "  const {points, cols, rows, cellW, cellH, canvasW, canvasH} = data;\n"
            "  const {imgAspect, meanDist} = data;\n"
            f"  const chrome = {PLOT_CHROME};  // px for summary + controls\n"
            "  // height follows the canvas aspect ratio\n"
            "  const height = Math.round(width * canvasH / canvasW) + chrome;\n"
            f"  return (\n{MORPH_JS.strip()}\n  );\n"
            "}\n"
        )

    def selector_js(load: str) -> str:
        """JS (statements) building a dropdown + chart; ``load(file)`` must return
        a promise of grid data. Expects ``grids`` ([{file, title}]), ``width``,
        ``invalidation`` in scope; leaves the DOM node in ``root``."""
        return f"""
      const root = document.createElement("div");
      const select = document.createElement("select");
      select.style.cssText = "font:14px sans-serif;margin-bottom:8px";
      for (const g of grids) select.append(new Option(g.title, g.file));
      const holder = document.createElement("div");
      root.append(select, holder);
      const cache = new Map();
      let stopCurrent = () => {{}};
      async function show(file) {{
        if (!cache.has(file)) cache.set(file, {load}(file));
        const data = await cache.get(file);
        if (select.value !== file) return;  // changed while loading
        stopCurrent();
        const stopped = new Promise((resolve) => (stopCurrent = resolve));
        holder.replaceChildren(renderGrid(data, width, stopped));
      }}
      select.onchange = () => show(select.value);
      invalidation.then(() => stopCurrent());
      show(select.value);
    """

    def write_web_files(out) -> list[dict]:
        """Regenerate the Observable cell and index.html for all grids in ``out``."""
        grids = []
        for f in sorted(out.glob("umap_grid_*.json")):
            with f.open() as fh:
                title = json.load(fh).get("title") or f.stem
            grids.append({"file": f.name, "title": title})

        # Observable: FileAttachment names must be string literals
        attachments = ",\n".join(
            f"    {json.dumps(g['file'])}: FileAttachment({json.dumps(g['file'])})"
            for g in grids
        )
        # a single anonymous expression (async IIFE) works as a cell in both
        # classic Observable notebooks and Notebooks 2.0 / Framework, unlike
        # the classic-only `name = { ... }` block syntax
        observable = (
            "// Observable cell. Attach these files with exactly these names:\n"
            + "".join(f"//   {g['file']}\n" for g in grids)
            + "(async () => {\n"
            f"  const grids = {json.dumps(grids)};\n"
            f"  const files = {{\n{attachments}\n  }};\n"
            + render_fn_js()
            + selector_js("((file) => files[file].json())")
            + "  return root;\n})()\n"
        )
        (out / "umap_grid_observable.js").write_text(observable)

        html = f"""<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>UMAP to grid</title>
    <style>body {{ margin: 0; font-family: sans-serif; }}</style>
    </head>
    <body>
    <div id="viz"></div>
    <script type="module">
    import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
    import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
    const grids = {json.dumps(grids)};
    const container = document.getElementById("viz");
    {render_fn_js()}
    let invalidate;
    function build() {{
      if (invalidate) invalidate();
      const invalidation = new Promise((resolve) => (invalidate = resolve));
      const width = container.clientWidth || window.innerWidth;
      const keep = container.querySelector("select")?.value;
      {selector_js("((file) => fetch(file).then((r) => r.json()))")}
      if (keep) {{ select.value = keep; show(keep); }}
      container.replaceChildren(root);
    }}
    build();
    let resizeTimer;
    window.addEventListener("resize", () => {{
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(build, 200);
    }});
    </script>
    </body>
    </html>
    """
        (out / "index.html").write_text(html)
        return grids

    return json, write_web_files


@app.cell
def _(
    Path,
    cell_h,
    cell_w,
    export_button,
    export_dir,
    export_name,
    export_title,
    grid_cols,
    grid_rows,
    img_aspect,
    json,
    layout,
    mean_dist,
    mo,
    points,
    write_web_files,
):
    import re

    mo.stop(not export_button.value)
    _name = re.sub(r"[^A-Za-z0-9_-]+", "-", export_name.value.strip()).strip("-")
    mo.stop(not _name, mo.md("**Enter a name for the export.**"))

    _out = Path(export_dir.value).expanduser()
    _out.mkdir(parents=True, exist_ok=True)
    _data = {
        "title": export_title.value.strip() or _name,
        # file name only; don't publish local paths
        "points": [{**p, "label": Path(p["label"]).name} for p in points],
        "cols": grid_cols,
        "rows": grid_rows,
        "cellW": cell_w,
        "cellH": cell_h,
        "canvasW": layout.canvas_width,
        "canvasH": layout.canvas_height,
        "imgAspect": round(img_aspect, 4),
        "meanDist": round(mean_dist, 4),
    }
    _json = json.dumps(_data, separators=(",", ":"))
    _file = _out / f"umap_grid_{_name}.json"
    _file.write_text(_json)
    _grids = write_web_files(_out)

    mo.md(
        f"Saved `{_file.name}` ({len(points)} items, {len(_json) / 1024:,.0f} KB). "
        f"Folder now has **{len(_grids)}** grid(s): "
        + ", ".join(f"{g['title']} (`{g['file']}`)" for g in _grids)
    )
    return


if __name__ == "__main__":
    app.run()
