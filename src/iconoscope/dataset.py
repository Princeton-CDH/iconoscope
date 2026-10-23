import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from PIL import Image

from iconoscope.models import EmbeddingConfig
from iconoscope.store import DatasetStore, StorageSchema
from iconoscope.umap import reduce_features

logger = logging.getLogger(__name__)

#: default image extensions
DEFAULT_IMG_EXTENSIONS = {".jpg", ".png", ".jpeg", ".tiff", ".webp", ".avif"}


def find_images(
    base_dir: Path,
    extensions: set[str] | None = None,
    max_images: int | None = None,
) -> Iterator[Path]:
    """
    Find images within the specified directory. Can optionally limit by file extension
    or stop when a specified maximum number of images is found.

    Results are yielded in filesystem order, which is arbitrary and varies by
    platform and filesystem (e.g., it may look sorted on macOS but not on Linux).
    When `max_images` is specified, the subset of images returned is therefore arbitrary
    and not guaranteed to be the same across systems; callers should not assume
    any particular order or subset.
    """

    # if no extensions are specified, use the defaults
    if extensions is None:
        extensions = DEFAULT_IMG_EXTENSIONS.copy()

    # because we want case-insensitive matching and will usually look for multiple
    # image extensions, by default look for any file extension and filter in the loop
    rglob_pattern = "*.*"
    single_ext = False
    # if  there is only one extension, use that as the pattern (loses the case-insensitivity)
    if len(extensions) == 1:
        rglob_pattern = f"*{list(extensions)[0]}"
        # ext-specific glob pattern means we can bypass the extra check
        single_ext = True

    found = 0
    for file_path in base_dir.rglob(rglob_pattern):
        # return if found by single extension or if suffix is in the list
        if single_ext or file_path.suffix.lower() in extensions:
            yield file_path
            found += 1

        # stop after the requested maximum if there is one
        if max_images is not None and found >= max_images:
            break


@dataclass
class ImageDataset:
    """
    Persistent image dataset, stored in an HDF5 file. Iterating yields
    loaded images and paths; see :mod:`iconoscope.embed` for use with a
    torch DataLoader (torch is not required to load or explore stored data).
    """

    #: path to existing hdf5 storage for image dataset; use :meth:`create`
    #: to create a new dataset
    storage_path: Path

    #: storage for this dataset, set from `storage_path` on init
    store: DatasetStore = field(init=False, repr=False)

    @classmethod
    def create(
        cls,
        image_dir: Path,
        storage_path: Path,
        *,
        extensions: set[str] | None = None,
        max_images: int | None = None,
    ) -> Self:
        """Collect images and create a new image dataset. Does not generate embeddings.

        Finds image files within the specified `image_dir` based on extension
        via :func:`find_images`. Image files are opened and verified to avoid
        adding broken image files to the dataset, and to determine image size.
        """
        # store image dir as absolute, so dataset is independent of working directory
        image_dir = Path(image_dir).resolve()
        storage_path = Path(storage_path)
        if not image_dir.is_dir():
            raise ValueError(f"Image directory `{image_dir}` is not a directory")
        if storage_path.exists():
            raise ValueError(f"Dataset already exists: {storage_path}")

        valid_paths = []
        image_sizes = []
        for path in find_images(
            image_dir, extensions=extensions, max_images=max_images
        ):
            try:
                with Image.open(path) as image:
                    image.verify()
                    # store paths relative to image dir; resolved on load
                    valid_paths.append(str(path.relative_to(image_dir)))
                    image_sizes.append(image.size)  # tuple[int] of width,height
            except (OSError, SyntaxError) as err:
                logger.warning("Skipping invalid image %s: %s", path, err)
                continue

        if not valid_paths:
            raise ValueError(f"No valid images found in `{image_dir}`")

        # glob order is indeterminate; sort by path before saving
        df = pl.DataFrame(data={"paths": valid_paths, "sizes": image_sizes}).sort(
            "paths"
        )
        DatasetStore.create(
            storage_path, image_dir, df["paths"].to_list(), df["sizes"].to_list()
        )

        return cls(storage_path=storage_path)

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)
        if not self.storage_path.exists():
            raise ValueError(
                f"Dataset `{self.storage_path}` does not exist; "
                "use ImageDataset.create() to create a new dataset"
            )
        self.store = DatasetStore(self.storage_path)

    @property
    def schema(self) -> type[StorageSchema]:
        """Group and dataset names within the hdf5 storage; from the store."""
        return self.store.schema

    def get_image_paths(self) -> Iterator[Path]:
        """Yield image paths from the stored dataset."""
        yield from self.store.image_paths()

    @property
    def image_count(self) -> int:
        """Return the number of paths in the persisted inventory."""
        return self.store.image_count

    ## iterable dataset logic

    def __iter__(self) -> Iterator[tuple[Image.Image, str]]:
        for file_path in self.get_image_paths():
            # may still need better error handling
            try:
                img = Image.open(file_path).convert("RGB")
            except OSError as err:
                logger.warning("Error loading %s: %s", file_path, err)
                continue
            # yield a tuple of image object and file path as string
            yield img, str(file_path)

    @staticmethod
    def collate(batch):
        # custom collate method for dataset collate_fn
        # image dataset batch is a list of tuples: [(img1, path1), (img2, path2), ...]
        # convert that to tuple of lists for this batch
        images, paths = zip(*batch)
        return list(images), list(paths)

    ## storage functionality

    def save_features(
        self,
        df: pl.DataFrame,
        config: EmbeddingConfig,
        overwrite: bool = False,
    ) -> None:
        """Save image features for the specified model; supports updating HDF5 without overwriting
        existing data, e.g. features from other models.

        If features for this model are already present, raises ValueError unless
        ``overwrite`` is True, in which case the model's entire group is replaced;
        any derived results stored for the model (e.g. umap, clusters) are removed,
        since they are no longer valid for the new features.
        """
        expected_columns = {"image_path", "features"}

        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}"
            )
        # image size and aspect ratio are allowed but not required (since now included in get_image_data)
        allowed_columns = {"image_size", "aspect_ratio"}
        unsupported_columns = set(df.columns) - expected_columns - allowed_columns
        if unsupported_columns:
            raise ValueError(
                f"unsupported columns: {', '.join(sorted(unsupported_columns))}"
            )

        features = df["features"].to_numpy()
        if features.ndim != 2:
            raise ValueError("features must be 2-dimensional")

        # single read/write open for validation, any pruning, and the write
        with self.store.open("r+") as store:
            # Image paths are shared across models and must match feature vectors.
            # Check that saved images match the current set before making any updates.
            img_df = store.image_data()
            # error if any image paths are in the incoming df but not in the stored image paths
            unknown_img_df = df.join(img_df, on="image_path", how="anti")
            if unknown_img_df.height:
                raise ValueError(
                    "image paths do not match the existing dataset "
                    f"({unknown_img_df.height} unknown paths)"
                )
            image_paths = df["image_path"]
            # if fewer images than stored (e.g. some failed to load), prune
            # the inventory; only allowed before any features are saved.
            # the store validates path order and uniqueness before modifying
            if img_df.height != df.height:
                store.prune_images(image_paths)

            store.write_features(config, features, image_paths, overwrite=overwrite)

    def has_features(self, config: EmbeddingConfig | None = None) -> bool:
        """Check if features are stored for the specified model + preprocessing
        configuration; if no configuration is specified, check for any features."""
        return self.store.has_features(config)

    def save_clusters(
        self,
        labels: np.ndarray,
        n_clusters: int,
        config: EmbeddingConfig,
        method: str = "kmeans",
    ) -> None:
        """Save cluster labels generated by clustering features for a
        configuration with the specified method and number of clusters.
        Results for different methods and numbers of clusters are kept
        separately."""
        self.store.write_clusters(config, labels, n_clusters, method=method)

    def info(self) -> dict:
        # single file open shared by all store reads below
        with self.store.open("r") as store:
            f = store.file
            # get top-level schema attributes and versions
            schema = f.attrs.get(self.schema.attr_schema)
            schema_version = f.attrs.get(self.schema.attr_schema_version)

            img_grp = f[self.schema.image_group]
            # image data includes aspect ratio
            img_df = store.image_data()

            info = {
                "image_paths": store.image_count,
                "image_dir": str(store.image_dir),
                # TODO: report number / % of images with mode size/aspect?
                "image_size_mode": img_df["image_size"].mode().to_list()[0],
                "image_aspect_mode": img_df["aspect_ratio"].mode().to_list()[0],
                "models": {},
            }
            if schema and schema_version:
                info["schema"] = f"{schema} {schema_version}"

            # stored configurations by group name, for config-based reads
            configs = {self.schema.model_group(c): c for c in store.configs()}
            all_models_grp = img_grp.get(self.schema.all_models_group)
            if all_models_grp:
                for model_grp_name in all_models_grp.keys():
                    model_grp = all_models_grp[model_grp_name]

                    # each model can have one or more group with preprocessing options
                    for model_opt_grp_name in model_grp.keys():
                        model_data = {}
                        model_opt_grp = model_grp[model_opt_grp_name]
                        features_dataset = model_opt_grp.get(self.schema.features_ds)
                        umap_dataset = model_opt_grp.get(self.schema.umap_ds)
                        if features_dataset is not None:
                            model_data["embeddings"] = features_dataset.shape
                        if umap_dataset is not None:
                            model_data["umap"] = umap_dataset.shape
                        # any validation ? check rows?

                        config = configs.get(f"{model_grp_name}/{model_opt_grp_name}")
                        cluster_info = store.cluster_info(config) if config else []
                        if cluster_info:
                            model_data["clusters"] = cluster_info
                        # for simplicity of reporting, report by model with option name as stored in the data file
                        info["models"][f"{model_grp_name}/{model_opt_grp_name}"] = (
                            model_data
                        )
            return info

    def get_configs(self) -> list[EmbeddingConfig]:
        """Embedding configurations (model + preprocessing options) with
        features stored in this dataset; see :meth:`DatasetStore.configs`."""
        return self.store.configs()

    def configs_by_label(self) -> dict[str, EmbeddingConfig]:
        """Stored embedding configurations keyed by display label
        (``model/key``), sorted by model name with default preprocessing first."""
        return {
            config.label: config
            for config in sorted(
                self.get_configs(),
                key=lambda c: (c.model.name, c.key != "defaults", c.key),
            )
        }

    def get_image_data(self) -> pl.DataFrame:
        """Image paths, sizes, and aspect ratios; see :meth:`DatasetStore.image_data`."""
        return self.store.image_data()

    def load_data(
        self,
        config: EmbeddingConfig | None = None,
        paths=True,
        features=False,
        umap=False,
        clusters: bool | int = False,
    ) -> pl.DataFrame:
        """Load image data, optionally with features, umap coordinates, and
        cluster labels for an embedding configuration. Umap coordinates are
        calculated and saved if not present.

        `clusters` may be True to include all stored cluster results, or a
        number of clusters to include only kmeans results for that k. Cluster
        labels are included as columns named ``cluster_{method}_{n_clusters}``.
        """
        # if umap is requested, open in read/write mode in case we need to save
        read_mode = "r+" if umap else "r"
        with self.store.open(read_mode) as store:
            # image paths, sizes, and aspect ratio from the store
            img_df = store.image_data() if paths else None
            data = {}

            if features or umap:
                # model configuration is required if features/umap requested
                if config is None:
                    raise ValueError(
                        "model configuration is required to load features or umap"
                    )

                feature_data = None
                if umap:
                    umap_coords = store.umap(config)
                    # if umap has not yet been generated, calculate and save
                    if umap_coords is None:
                        logger.info(
                            "UMAP not present for %s; calculating", config.label
                        )
                        feature_data = store.features(config)
                        umap_coords = reduce_features(feature_data)
                        store.write_umap(config, umap_coords)
                    else:
                        logger.debug("Loading saved UMAP for %s", config.label)
                    data["umap"] = umap_coords

                if features:
                    # load features if not already loaded for umap
                    if feature_data is None:
                        feature_data = store.features(config)
                    data["features"] = feature_data

            # include cluster labels when requested and present in the data
            # (bool is a subclass of int, so check for it explicitly)
            if clusters is not False:
                if config is None:
                    raise ValueError("model configuration is required to load clusters")
                n_clusters = None if clusters is True else clusters
                for (method, k), labels in store.clusters(
                    config, n_clusters=n_clusters
                ).items():
                    data[f"cluster_{method}_{k}"] = labels

        df = pl.DataFrame(data=data)
        if img_df is not None:
            df = pl.concat([img_df, df], how="horizontal") if data else img_df
        return df
