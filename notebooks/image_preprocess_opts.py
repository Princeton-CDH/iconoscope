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


if __name__ == "__main__":
    app.run()
