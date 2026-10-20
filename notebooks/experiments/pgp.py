import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from iconoscope.dataset import ImageDataset

    return ImageDataset, Path, mo


@app.cell
def _(ImageDataset, Path):
    pgp_dataset_path = Path(
        "/Users/rkoeser/workarea/experiments/20260608-pgp-vislab/pgp.h5"
    )
    img_dataset = ImageDataset(pgp_dataset_path)
    return (img_dataset,)


@app.cell
def _(img_dataset):
    img_dataset.get_configs()
    return


@app.cell
def _(img_dataset):
    img_dataset.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## umap/mosaic
    """)
    return


@app.cell
def _(img_dataset, mo):
    _configs = img_dataset.configs_by_label()
    embedding_select = mo.ui.dropdown(
        options=_configs, value=list(_configs.keys())[0], label="Embedding"
    )
    embedding_select
    return


if __name__ == "__main__":
    app.run()
