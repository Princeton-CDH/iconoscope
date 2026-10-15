import math
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tqdm
from lap import lapjv
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from iconoscope.dataset import ImageDataset
from iconoscope.models import EmbeddingModel

LAPJV_CELL_LIMIT = 5000


def assign_to_grid(
    coords: np.ndarray, grid_cols: int, grid_rows: int
) -> dict[tuple[int, int], int]:
    """Given an array of x,y coordinates as returned from :meth:`reduce_features` and
    a grid size based on columns and rows, assign each item in the coordinates
    array to a cell in the grid. Uses lapjv (Jonker-Volgenant) to determine the best fit
    for grids up to LAPJV_CELL_LIMIT cells; falls back to greedy KD-tree for larger grids.

    Returns a dictionary of the grid assignment for each input item, based on index
    in the original coords array.
        {(row, col): item_idx}. Cells with no image are omitted.
    """

    # determine number of items to be positioned based on number of x,y coordinates
    n_items = len(coords)
    # determine number of cells in the grid
    n_cells = grid_cols * grid_rows

    # warn if grid size results in any images being omitted
    if n_items > n_cells:
        print(f"Warning: omitting {n_items - n_cells:,} images from the grid")

    # generate an array of grid coordinates for the requested grid size
    grid_cells = np.array(
        [(c, r) for r in range(grid_rows) for c in range(grid_cols)],
        dtype=np.float32,
    )
    # Keep grid centers in the same x,y coordinate order as the input coordinates.
    cell_centers = (grid_cells + 0.5) / np.array(
        [[grid_cols, grid_rows]], dtype=np.float32
    )
    # determine aspect ratio for the requested grid
    aspect = grid_cols / grid_rows
    # scale the cell center coordinates based on the aspect ratio of the requested grid
    scale = np.array([[aspect, 1.0]], dtype=np.float32)

    if n_cells <= LAPJV_CELL_LIMIT:
        # lapjv requires a square cost matrix; if there are more cells than items,
        # add padding items to fill out the grid
        if n_items < n_cells:
            # put the padding cells at bottom right instead of middle
            padding = np.full((n_cells - n_items, 2), 1.0, dtype=np.float32)
            square_coords = np.vstack([coords, padding])
        else:
            square_coords = coords[:n_cells]

        # compute pairwise Euclidean distances to create cost matrix for lapjv
        cost = cdist(square_coords * scale, cell_centers * scale).astype(np.float64)
        # lap.lapjv returns a tuple of optional cost, reverse mapping, and column index.
        # The column index is the assigned mapping: array of item indices assigned to each cell
        _, _, col_ind = lapjv(cost)
        # use lapjv column index to map back to row/column placement in the grid
        return {
            # take lapjv assigned slot for each image and map to grid position
            # decompose the flat array of x,y grid cells back into row,col format
            (
                int(grid_cells[cell_idx][1]),
                int(grid_cells[cell_idx][0]),
            ): item_idx
            for cell_idx, item_idx in enumerate(col_ind)
            if item_idx < n_items  # omit any padding items needed to make square
        }

    else:
        # build spatial index of the grid cells for quick lookup of nearest cells
        tree = cKDTree(cell_centers * scale)
        # scale image coordinates to the grid; omit any beyond number of available slots
        # NOTE: important to truncate coords to n_cells or loop will never finish
        scaled_coords = coords[:n_cells] * scale
        # keep track of grid cells as they are claimed
        used: set[int] = set()
        # return structure, same as for lapjv: tuple of row,col in grid and image index
        assignments: dict[tuple[int, int], int] = {}
        k = min(n_cells, 50)
        for item_idx in range(len(scaled_coords)):
            # query for the k nearest cells
            while True:
                _, cell_indices = tree.query(scaled_coords[item_idx], k=k)
                # choose the first cell that is not already used
                chosen = next((i for i in cell_indices if i not in used), None)
                # stop looping as soon as grid cell is chosen
                if chosen is not None:
                    break
                # if nearest cells are already taken, expand the search - quadruple k and try again
                k = min(k * 4, n_cells)
            used.add(chosen)
            # add the chosen grid x,y as row,col integers for the assignment
            assignments[(int(grid_cells[chosen][1]), int(grid_cells[chosen][0]))] = (
                item_idx
            )
        return assignments


MIN_THUMB_SIZE = 20
MAX_THUMB_SIZE = 350


def estimate_columns(image_count, canvas_width, canvas_height, aspect_ratio) -> int:
    """
    Estimate of the best column count for the requested canvas size, given
    the number of images with the specified aspect ratio.

    aspect_ratio = thumbnail width / thumbnail height

    Treat each image as a cell with the desired aspect ratio,
    require the total area of all cells to equal the area of
    the full canvas, and solve for thumbnail width.

        image_count * w * (w / aspect_ratio)
            = canvas_width * canvas_height

    Solving for thumbnail width w:
        w = sqrt(canvas_width * canvas_height * aspect_ratio / image_count)
    """
    canvas_area = canvas_width * canvas_height
    thumb_width = math.sqrt(canvas_area * aspect_ratio / image_count)

    return round(canvas_width / thumb_width)


class GridLayout(NamedTuple):
    """Grid size chosen by :func:`best_grid`.

    ``thumb_width`` and ``thumb_height`` are the largest thumbnail size that
    fits the canvas while preserving the requested aspect ratio exactly. When
    the grid is stretched to fill the canvas (as :func:`generate_mosaic` does),
    the actual cell size is ``canvas_width / cols`` by ``canvas_height / rows``,
    which may differ from this ideal size in one dimension.
    """

    rows: int
    cols: int
    #: ideal thumbnail width for the requested aspect ratio (not stretched)
    thumb_width: float
    #: ideal thumbnail height for the requested aspect ratio (not stretched)
    thumb_height: float


def best_grid(
    image_count, canvas_width, canvas_height, aspect_ratio, search_window=3
) -> GridLayout:
    """Choose the grid size near the estimated column count that allows the
    largest thumbnails at the requested aspect ratio.

    Returns a :class:`GridLayout` with rows, cols, and the ideal thumbnail
    size for the aspect ratio. Callers that fill the canvas should compute
    cell size from the canvas size and rows/cols instead."""
    estimated_cols = estimate_columns(
        image_count, canvas_width, canvas_height, aspect_ratio
    )

    best = None
    first_col = max(1, estimated_cols - search_window)
    last_col = estimated_cols + search_window

    for cols in range(first_col, last_col + 1):
        rows = math.ceil(image_count / cols)

        max_cell_width = canvas_width / cols
        max_cell_height = canvas_height / rows

        # Thumbnail must fit the cell in both directions.
        thumb_width = min(max_cell_width, max_cell_height * aspect_ratio)

        if best is None or thumb_width > best.thumb_width:
            best = GridLayout(rows, cols, thumb_width, thumb_width / aspect_ratio)

    return best


class MosaicLayout(NamedTuple):
    """Full mosaic layout computed by :func:`layout_mosaic`: grid size,
    the (integer pixel) cell size used to fill the canvas, and grid assignments."""

    grid: GridLayout
    #: cell width in pixels (grid stretched to fill the requested canvas)
    cell_width: int
    #: cell height in pixels
    cell_height: int
    #: {(row, col): item_idx}, as returned by :func:`assign_to_grid`
    assignments: dict[tuple[int, int], int]

    @property
    def canvas_width(self) -> int:
        """Actual canvas width; may differ slightly from requested due to rounding."""
        return self.grid.cols * self.cell_width

    @property
    def canvas_height(self) -> int:
        """Actual canvas height; may differ slightly from requested due to rounding."""
        return self.grid.rows * self.cell_height


def layout_mosaic(
    coords: np.ndarray, canvas_width: int, canvas_height: int, aspect_ratio: float
) -> MosaicLayout:
    """Determine grid size and cell size for the canvas and image aspect ratio,
    and assign each 2D coordinate (normalized 0-1) to a grid cell."""
    grid = best_grid(len(coords), canvas_width, canvas_height, aspect_ratio)
    # determine thumbnail size that will fill the canvas
    cell_width = round(canvas_width / grid.cols)
    cell_height = round(canvas_height / grid.rows)
    assignments = assign_to_grid(coords, grid.cols, grid.rows)
    return MosaicLayout(grid, cell_width, cell_height, assignments)


def load_thumbnail(path: str | Path, size: tuple[int, int]) -> Image.Image:
    """Open an image and resize to exactly ``size`` (width, height), cropping
    as needed to fill the space while preserving the image aspect ratio."""
    with Image.open(path) as img:
        # NOTE: ImageOps.cover only scales (result may be larger than size);
        # fit scales and center-crops to exactly size
        return ImageOps.fit(img.convert("RGB"), size, Image.LANCZOS)


def generate_mosaic(
    img_dataset: ImageDataset,
    model: EmbeddingModel,
    output: Path | None = None,
    width: int = 2000,
    height: int = 2000,
    jpeg_quality: int = 90,
    sample_size: int | None = None,
) -> None:
    """Generate a mosaic of images based on previously calculated image embeddings (feature vectors).
    Load image feature vectors from the specified image dataset file, get or generate 2D coordinates
    via PCA+UMAP (saved in the dataset file), then assign to grid slots and create
    a mosaic image of thumbnails.
    """

    if output is None:
        output = img_dataset.storage_path.with_suffix(".jpg")

    # load image embeddings from hdf5 file for specified model
    df = img_dataset.load_data(model=model, umap=True)

    # if a sample is requested, select a random sample of the specified size
    # (subset after UMAP coords are generated, since they should be done for all images)
    if sample_size is not None:
        df = df.sample(sample_size)

    paths = df["image_path"].to_list()

    # determine ideal thumbnail size based on mosaic size and number of images
    n_images = df.height
    # use the most frequent image aspect ratio in the dataset as thumbnail aspect ratio
    img_aspect_ratio = df["aspect_ratio"].mode()[0]

    layout = layout_mosaic(df["umap"].to_numpy(), width, height, img_aspect_ratio)
    thumb_size = (layout.cell_width, layout.cell_height)

    # create a blank canvas for the calculated size
    # (due to rounding, actual canvas size may not be exactly as requested)
    canvas = Image.new(
        "RGB", (layout.canvas_width, layout.canvas_height), color=(0, 0, 0)
    )
    # using black bg instead of white, which would be color=(255, 255, 255)
    for (row, col), img_idx in tqdm.tqdm(
        layout.assignments.items(), total=n_images, desc="Creating mosaic"
    ):
        try:
            thumb = load_thumbnail(paths[img_idx], thumb_size)
            # paste the thumbnail on the grid in the appropriate slot
            canvas.paste(thumb, (col * layout.cell_width, row * layout.cell_height))
        except Exception as exc:
            warnings.warn(f"Could not load {paths[img_idx]}: {exc}")

    canvas.save(output, quality=jpeg_quality)
    print(f"Saved mosaic to {output}")
