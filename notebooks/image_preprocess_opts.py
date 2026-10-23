import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import transformers

    from iconoscope.models import AVAILABLE_MODELS

    return AVAILABLE_MODELS, mo, transformers


@app.cell
def _(AVAILABLE_MODELS, mo, transformers):
    output = []

    for model in AVAILABLE_MODELS.values():
        output.append(f"## {model.name}")
        processor = transformers.AutoImageProcessor.from_pretrained(model.id)
        output.append(f"```\n{str(processor)}\n```\n")

    mo.md("\n".join(output))
    return


@app.cell
def _(AVAILABLE_MODELS, mo):
    mo.md("## Preview model input")
    img_dir = mo.ui.text(label="Image directory", full_width=True)
    num_images = mo.ui.slider(1, 12, value=4, label="Images", show_value=True)
    model_select = mo.ui.dropdown(
        options=list(AVAILABLE_MODELS.keys()),
        value=list(AVAILABLE_MODELS.keys())[0],
        label="Model",
    )
    use_crop = mo.ui.checkbox(label="Center crop")
    crop_pct = mo.ui.slider(
        0.1, 1.0, step=0.05, value=0.6, label="Crop fraction", show_value=True
    )
    grayscale = mo.ui.checkbox(label="Grayscale")
    mo.vstack(
        [img_dir, num_images, model_select, mo.hstack([use_crop, crop_pct]), grayscale]
    )
    return crop_pct, grayscale, img_dir, model_select, num_images, use_crop


@app.cell
def _(
    AVAILABLE_MODELS,
    crop_pct,
    grayscale,
    img_dir,
    mo,
    model_select,
    num_images,
    transformers,
    use_crop,
):
    from pathlib import Path

    from PIL import Image

    from iconoscope.dataset import find_images
    from iconoscope.embed import preview_model_input
    from iconoscope.models import EmbeddingConfig

    mo.stop(not img_dir.value, mo.md("*Enter an image directory to preview.*"))

    _paths = sorted(
        find_images(Path(img_dir.value).expanduser(), max_images=num_images.value)
    )
    _originals = [Image.open(p).convert("RGB") for p in _paths]
    _model = AVAILABLE_MODELS[model_select.value]
    _processor = transformers.AutoImageProcessor.from_pretrained(_model.id)
    _defaults = preview_model_input(_originals, EmbeddingConfig(_model), _processor)
    _custom_config = EmbeddingConfig(
        _model,
        center_crop=crop_pct.value if use_crop.value else None,
        grayscale=grayscale.value,
    )
    _custom = preview_model_input(_originals, _custom_config, _processor)

    def _thumb(img):
        _t = img.copy()
        _t.thumbnail((224, 224))
        return mo.image(_t)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([mo.md(f"**{p.name}** (original)"), _thumb(o)]),
                    mo.vstack([mo.md("model defaults"), mo.image(d)]),
                    mo.vstack([mo.md(f"`{_custom_config.key}`"), mo.image(c)]),
                ],
                justify="start",
            )
            for p, o, d, c in zip(_paths, _originals, _defaults, _custom)
        ]
    )
    return


if __name__ == "__main__":
    app.run()
