import numpy as np
import pytest

from iconoscope import mosaic


@pytest.mark.parametrize("grid_cols, grid_rows", [(2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("use_greedy", [False, True])
def test_assign_to_grid_preserves_xy_orientation(
    monkeypatch: pytest.MonkeyPatch,
    grid_cols: int,
    grid_rows: int,
    use_greedy: bool,
) -> None:
    """Coordinates are x/y, while assignments are returned as row/column."""
    if use_greedy:
        monkeypatch.setattr(mosaic, "LAPJV_CELL_LIMIT", 0)

    coords = np.array(
        [
            (col + 0.5, row + 0.5)
            for row in range(grid_rows)
            for col in range(grid_cols)
        ],
        dtype=np.float32,
    )
    coords /= (grid_cols, grid_rows)

    assignments = mosaic.assign_to_grid(coords, grid_cols, grid_rows)

    assert assignments == {
        (row, col): row * grid_cols + col
        for row in range(grid_rows)
        for col in range(grid_cols)
    }


@pytest.mark.parametrize(
    "image_count, canvas_width, canvas_height, aspect_ratio, expected_cols",
    [
        # square canvas, square thumbnails: 10x10 grid of 100px thumbs
        (100, 1000, 1000, 1.0, 10),
        # wide canvas, square thumbnails: 20x10 grid of 100px thumbs
        (200, 2000, 1000, 1.0, 20),
        # tall canvas, square thumbnails: 10x20 grid of 100px thumbs
        (200, 1000, 2000, 1.0, 10),
        # landscape thumbnails (200x100): 5 cols x 10 rows
        (50, 1000, 1000, 2.0, 5),
        # portrait thumbnails (50x100): 20 cols x 10 rows
        (200, 1000, 1000, 0.5, 20),
        # single image fills the canvas
        (1, 1000, 1000, 1.0, 1),
    ],
)
def test_estimate_columns(
    image_count: int,
    canvas_width: int,
    canvas_height: int,
    aspect_ratio: float,
    expected_cols: int,
) -> None:
    """Total thumbnail area should match canvas area for exact-fit cases."""
    assert (
        mosaic.estimate_columns(image_count, canvas_width, canvas_height, aspect_ratio)
        == expected_cols
    )


def test_estimate_columns_increases_with_image_count() -> None:
    """More images on the same canvas require more (smaller) columns."""
    cols = [mosaic.estimate_columns(n, 1000, 1000, 1.0) for n in (10, 100, 1000)]
    assert cols == sorted(cols)
    assert cols[0] < cols[-1]


@pytest.mark.parametrize(
    "image_count, canvas_width, canvas_height, aspect_ratio, expected",
    [
        (100, 1000, 1000, 1.0, (10, 10)),
        (200, 2000, 1000, 1.0, (10, 20)),
        (50, 1000, 1000, 2.0, (10, 5)),
    ],
)
def test_best_grid(
    image_count: int,
    canvas_width: int,
    canvas_height: int,
    aspect_ratio: float,
    expected: tuple[int, int],
) -> None:
    layout = mosaic.best_grid(image_count, canvas_width, canvas_height, aspect_ratio)
    assert (layout.rows, layout.cols) == expected
    assert layout.rows * layout.cols >= image_count
    # ideal thumbnail size preserves aspect ratio and fits within the canvas
    assert layout.thumb_width / layout.thumb_height == pytest.approx(aspect_ratio)
    assert layout.thumb_width * layout.cols <= canvas_width
    assert layout.thumb_height * layout.rows <= canvas_height


@pytest.mark.parametrize(
    ("image_count", "canvas_width", "canvas_height", "aspect_ratio"),
    [(100, 2000, 2000, 1.0), (50, 2000, 1200, 0.75), (37, 1000, 1000, 1.5)],
)
def test_layout_mosaic(
    image_count: int, canvas_width: int, canvas_height: int, aspect_ratio: float
) -> None:
    coords = np.random.default_rng(0).random((image_count, 2)).astype(np.float32)
    layout = mosaic.layout_mosaic(coords, canvas_width, canvas_height, aspect_ratio)
    grid = mosaic.best_grid(image_count, canvas_width, canvas_height, aspect_ratio)
    assert layout.grid == grid
    # cells stretch the grid to fill the canvas
    assert layout.cell_width == round(canvas_width / grid.cols)
    assert layout.cell_height == round(canvas_height / grid.rows)
    assert layout.canvas_width == grid.cols * layout.cell_width
    assert layout.canvas_height == grid.rows * layout.cell_height
    # every image placed exactly once, in a valid cell
    assert sorted(layout.assignments.values()) == list(range(image_count))
    for row, col in layout.assignments:
        assert 0 <= row < grid.rows and 0 <= col < grid.cols


def test_load_thumbnail(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "wide.png"
    Image.new("RGB", (400, 100), (255, 0, 0)).save(path)
    thumb = mosaic.load_thumbnail(path, (30, 40))
    assert thumb.size == (30, 40)
    assert thumb.mode == "RGB"


def test_render_mosaic(tmp_path) -> None:
    from PIL import Image

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    paths = []
    for i, color in enumerate(colors):
        path = tmp_path / f"img{i}.png"
        Image.new("RGB", (40, 30), color).save(path)
        paths.append(path)
    # one image per corner
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    layout = mosaic.layout_mosaic(coords, 200, 100, 40 / 30)
    canvas = mosaic.render_mosaic(paths, coords, 40 / 30, 200, 100)
    assert canvas.size == (layout.canvas_width, layout.canvas_height)
    # each image is pasted in its assigned cell
    for (row, col), idx in layout.assignments.items():
        center = (
            int((col + 0.5) * layout.cell_width),
            int((row + 0.5) * layout.cell_height),
        )
        assert canvas.getpixel(center) == colors[idx]


def test_render_mosaic_missing_image(tmp_path) -> None:
    coords = np.array([[0.5, 0.5]], dtype=np.float32)
    with pytest.warns(UserWarning, match="Could not load"):
        canvas = mosaic.render_mosaic([tmp_path / "missing.png"], coords, 1.0, 50, 50)
    # left blank (black)
    assert canvas.getpixel((10, 10)) == (0, 0, 0)


@pytest.fixture
def mock_dataset(tmp_path):
    """Mock ImageDataset with 5 images; returns real polars data from load_data."""
    from unittest.mock import MagicMock

    import polars as pl

    df = pl.DataFrame(
        {
            "image_path": [str(tmp_path / f"img{i}.png") for i in range(5)],
            "umap": np.random.default_rng(0).random((5, 2), dtype=np.float32),
            "aspect_ratio": [1.5] * 5,
        }
    )
    dataset = MagicMock(storage_path=tmp_path / "data.h5")
    dataset.load_data.return_value = df
    return dataset


@pytest.mark.parametrize(
    "sample_size, expected_count",
    [(None, 5), (3, 3), (4, 4), (5, 5), (6, 5), (1000, 5)],
)
def test_generate_mosaic_sample_size(
    mock_dataset, tmp_path, sample_size, expected_count
) -> None:
    from unittest.mock import patch

    with patch("iconoscope.mosaic.render_mosaic") as render:
        mosaic.generate_mosaic(
            mock_dataset,
            config=None,
            output=tmp_path / "out.jpg",
            sample_size=sample_size,
        )
    paths, coords, *_ = render.call_args.args
    assert len(paths) == expected_count
    assert len(coords) == expected_count
    all_paths = set(mock_dataset.load_data.return_value["image_path"])
    assert set(paths) <= all_paths
    render.return_value.save.assert_called_once_with(tmp_path / "out.jpg", quality=90)
