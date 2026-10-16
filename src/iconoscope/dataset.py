import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import polars as pl
from PIL import Image
from torch.utils.data import IterableDataset

from iconoscope.models import AVAILABLE_MODELS, EmbeddingConfig
from iconoscope.umap import reduce_features

logger = logging.getLogger(__name__)

#: default image extensions
DEFAULT_IMG_EXTENSIONS = {".jpg", ".png", ".jpeg", ".tiff", ".webp", ".avif"}

# base class for iterable dataset without storage?


def find_images(
    base_dir: Path,
    extensions: set[str] | None = None,
    max: int | None = None,
) -> Iterator[Path]:
    """
    Find images within the specified directory. Can optionally limit by file extension
    or stop when a specified maximum number of images is found.

    Results are yielded in filesystem order, which is arbitrary and varies by
    platform and filesystem (e.g., it may look sorted on macOS but not on Linux).
    When `max` is specified, the subset of images returned is therefore arbitrary
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
        if max is not None and found >= max:
            break


class StorageSchema:
    """
    Group and dataset names for iconoscope image dataset within HDF5 file.
    """

    # schema versioning
    schema_name: str = "iconoscope"
    version: float = 0.2

    image_group: str = "image"
    # relative to image group
    image_ds_paths: str = "paths"
    image_ds_sizes: str = "sizes"
    all_models_group: str = "models"
    # relative to model group©
    features_ds: str = "features"
    umap_ds: str = "umap"
    cluster_ds: str = "cluster"

    # model group attrs
    attr_model = "model"
    attr_model_id = "model_id"

    # schema attrs
    attr_schema = "schema"
    attr_schema_version = "schema_version"

    @classmethod
    def model_group(cls, config: EmbeddingConfig, absolute: bool = False) -> str:
        """Group path for features and derived data for a model + preprocessing
        configuration; relative to the all-models group unless `absolute`."""
        base_path = f"{cls.image_group}/{cls.all_models_group}/" if absolute else ""
        return f"{base_path}{config.model.name}/{config.key}"


# IMAGE_GROUP = "image"
# PATHS = "paths"
# MODELS = "models"
# FEATURES = "features"
# UMAP = "umap"
# CLUSTER = "cluster"
# IMAGE_DIR_ATTR = "image_dir"
# N_CLUSTERS_ATTR = "n_clusters"


@dataclass
class ImageDataset(IterableDataset):
    """
    Persistent image dataset. Provides iterable image dataset logic for use with torch DataLoader,
    but also provides logic for saving and loading from HDF5 file.
    """

    #: path to hdf5 storage for image dataset
    storage_path: Path

    #: base directory for images in this dataset - OPTIONAL (saved as an attr in hf5 on creation)
    image_dir: Path | None = None

    #: image extensions; if not specified, uses the defaults
    extensions: set[str] = field(default_factory=DEFAULT_IMG_EXTENSIONS.copy)

    #: optional limit for number of images to find; which images are included
    #: is arbitrary (filesystem order), see :func:`find_images`
    max_images: int | None = None

    #: schema for group and dataset names within the hdf5 storage
    schema: StorageSchema = field(default_factory=StorageSchema)

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
        for path in find_images(image_dir, extensions=extensions, max=max_images):
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

        # glob order is indeterminate; load into a dataframe and sort by path before saving
        df = pl.DataFrame(data={"paths": valid_paths, "sizes": image_sizes}).sort(
            "paths"
        )

        # TODO: move somewhere reusable
        schema = StorageSchema()
        with h5py.File(storage_path, "w") as h5_file:
            # store iconoscope schema & version in top-level metadata
            h5_file.attrs[schema.attr_schema] = schema.schema_name
            h5_file.attrs[schema.attr_schema_version] = schema.version
            image_group = h5_file.create_group(schema.image_group)
            # image paths are stored relative to the (absolute) image dir
            image_group.attrs["image_dir"] = str(image_dir)
            image_group.create_dataset(
                schema.image_ds_paths,
                data=df["paths"].to_list(),
                compression="gzip",
            )
            image_group.create_dataset(
                schema.image_ds_sizes,
                data=df["sizes"].to_list(),
                compression="gzip",
            )

        return cls(storage_path=storage_path)

    def __post_init__(self):
        # when creating a new collection, storage will not exist so image dir is required
        # TODO: require create with image dir, and only open existing storage in init method
        if not self.storage_path.exists():
            if self.image_dir is None:
                raise ValueError(
                    f"image_dir is required when creating new image dataset ({self.storage_path} does not exist)"
                )
            elif not self.image_dir.is_dir():
                raise ValueError(
                    f"Image directory `{self.image_dir}` is not a directory"
                )
        # otherwise, storage exists and we will load paths from it

    def get_image_paths(self) -> Iterator[Path]:
        """Yield a list of image paths either from configured image directory or stored data file"""
        # TODO: simplify when we update init to storage only
        if self.storage_path.exists():
            for row in self.get_image_data().iter_rows(named=True):
                yield Path(row["image_path"])
        elif self.image_dir:
            # print("finding images on disk")
            yield from find_images(self.image_dir, max=self.max_images)

    @property
    def image_count(self) -> int:
        """Return the number of paths in the persisted inventory."""
        with h5py.File(self.storage_path, "r") as h5_file:
            return h5_file["image/paths"].size

    ## iterable dataset logic

    def __iter__(self) -> Iterator[tuple[Image.Image, str]]:
        for file_path in self.get_image_paths():
            # may still need better error handling
            try:
                img = Image.open(file_path).convert("RGB")
            except OSError as err:
                logger.warning(f"Error loading {file_path}: {err}")
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

        image_paths = df["image_path"].to_numpy()
        features = df["features"].to_numpy()
        if features.ndim != 2:
            raise ValueError("features must be 2-dimensional")

        # open as append so we can update without overwriting
        with h5py.File(self.storage_path, "a") as f:
            # get group for image information; create it if needed
            img_grp = f.require_group(self.schema.image_group)

            # Image paths are shared across models and must match feature vectors.
            # Check that saved images match the current set before making any updates.
            # TODO: update this once we split out dataset creation from embed
            if self.schema.image_ds_paths in img_grp:
                img_df = self.get_image_data()
                # error if any image paths are in the incoming df but not in the stored image paths
                unknown_img_df = df.join(img_df, on="image_path", how="anti")
                if unknown_img_df.height:
                    raise ValueError(
                        "image paths do not match the existing dataset "
                        f"({unknown_img_df.height} unknown paths)"
                    )
                # check if total does not match
                if img_df.height != df.height:
                    # if any features have already been saved, error
                    has_model_features = len(
                        img_grp.get(self.schema.all_models_group, [])
                    )
                    if has_model_features:
                        raise ValueError(
                            "cannot prune image paths when existing models are present"
                        )
                    else:
                        # otherwise, prune invalid image paths and their corresponding sizes
                        img_subset_df = df.join(img_df, on="image_path", how="left")
                        # store paths relative to image dir
                        base = Path(img_grp.attrs["image_dir"])
                        subset_paths = [
                            str(Path(p).relative_to(base))
                            for p in img_subset_df["image_path"]
                        ]
                        # can't resize, so delete and recreate
                        # TODO: move this somewhere shared?
                        del img_grp[self.schema.image_ds_paths]
                        del img_grp[self.schema.image_ds_sizes]
                        img_grp.create_dataset(
                            self.schema.image_ds_paths,
                            data=subset_paths,
                            compression="gzip",
                        )
                        img_grp.create_dataset(
                            self.schema.image_ds_sizes,
                            data=img_subset_df["image_size"].to_list(),
                            compression="gzip",
                        )

            # Features for a model already saved to this file are an error
            # unless overwrite is requested.
            all_models_grp = img_grp.require_group(self.schema.all_models_group)
            # determine group name for current model with current options
            model_group_name = self.schema.model_group(config)
            if model_group_name in all_models_grp:
                if not overwrite:
                    raise ValueError(
                        f"Features for '{model_group_name}' are already present and overwrite not requested."
                    )
                # delete the whole model group, since any downstream results
                # (umap/cluster) will be invalid for the new features
                del all_models_grp[model_group_name]

            # Save shared metadata and create the new model only after validation.
            # TODO: also save max if specified and extensions if not default
            if self.schema.image_ds_paths not in img_grp:
                # dataset not created via create(); store image dir as absolute
                # and paths relative to it, same as create()
                image_dir = Path(self.image_dir).resolve()
                img_grp.attrs["image_dir"] = str(image_dir)
                img_grp.create_dataset(
                    self.schema.image_ds_paths,
                    data=[
                        str(Path(p).resolve().relative_to(image_dir))
                        for p in image_paths
                    ],
                    compression="gzip",
                )
            model_grp = all_models_grp.create_group(model_group_name)
            # store model + preprocessing options so config can be reconstructed
            model_grp.attrs[self.schema.attr_model] = config.model.name
            model_grp.attrs[self.schema.attr_model_id] = config.model.id
            for opt, value in config.opts.items():
                # HDF5 attrs can't store None; omit unset options
                if value is not None:
                    model_grp.attrs[opt] = value
            # save extracted features as a dataset
            model_grp.create_dataset(
                self.schema.features_ds, data=features, compression="gzip"
            )

    def has_features(self, config: EmbeddingConfig | None = None) -> bool:
        """Check if features are stored for the specified model + preprocessing
        configuration; if no configuration is specified, check for any features."""
        if not self.storage_path.exists():
            return False
        with h5py.File(self.storage_path, "r") as f:
            if config is not None:
                model_group_name = self.schema.model_group(config, absolute=True)
                return (
                    model_group_name in f
                    and self.schema.features_ds in f[model_group_name]
                )
            all_models_grp = f.get(
                f"{self.schema.image_group}/{self.schema.all_models_group}"
            )
            if all_models_grp is None:
                return False
            # check for features stored under any model/options group
            return any(
                self.schema.features_ds in opt_grp
                for model_grp in all_models_grp.values()
                for opt_grp in model_grp.values()
            )

    def save_clusters(
        self,
        labels: np.ndarray,
        n_clusters: int,
        config: EmbeddingConfig,
    ) -> None:
        # save cluster labels generated by clustering features from a specific model
        with h5py.File(self.storage_path, "r+") as f:
            model_group_name = self.schema.model_group(config, absolute=True)
            model_grp = f[model_group_name]
            # create new dataset and save labels
            cluster_ds = model_grp.create_dataset(
                self.schema.cluster_ds, data=labels, compression="gzip"
            )
            # save the number of clusters in an attribute
            cluster_ds.attrs["n_clusters"] = n_clusters

    def info(self) -> dict:
        image_count = self.image_count
        with h5py.File(self.storage_path, "r") as f:
            # get top-level schema attributes and versions
            schema = f.attrs.get(self.schema.attr_schema)
            schema_version = f.attrs.get(self.schema.attr_schema_version)

            img_grp = f[self.schema.image_group]
            orig_img_dir = img_grp.attrs.get("image_dir")

            img_df = self.get_image_data().with_columns(
                aspect_ratio=pl.col.image_size.arr.first().truediv(
                    pl.col.image_size.arr.last()
                )
            )

            info = {
                "image_paths": image_count,
                "image_dir": orig_img_dir,
                # TODO: report number / % of images with mode size/aspect?
                "image_size_mode": img_df["image_size"].mode().to_list()[0],
                "image_aspect_mode": img_df["aspect_ratio"].mode().to_list()[0],
                "models": {},
            }
            if schema and schema_version:
                info["schema"] = f"{schema} {schema_version}"

            all_models_grp = img_grp.get(self.schema.all_models_group)
            if all_models_grp:
                for model_grp_name in all_models_grp.keys():
                    model_grp = all_models_grp[model_grp_name]

                    # TODO: update to report multiple features for model with alt preprocessing
                    # each model can have one or more group with preprocessing options
                    for model_opt_grp_name in model_grp.keys():
                        model_data = {}
                        model_opt_grp = model_grp[model_opt_grp_name]
                        features_dataset = model_opt_grp.get(self.schema.features_ds)
                        umap_dataset = model_opt_grp.get(self.schema.umap_ds)
                        cluster_dataset = model_opt_grp.get(self.schema.cluster_ds)
                        if features_dataset is not None:
                            model_data["embeddings"] = features_dataset.shape
                        if umap_dataset is not None:
                            model_data["umap"] = umap_dataset.shape
                        if cluster_dataset is not None:
                            model_data["cluster"] = {
                                "k": cluster_dataset.attrs.get("n_clusters"),
                                "size": cluster_dataset.size,
                            }
                        # any validation ? check rows?

                        # for simplicity of reporting, report by model with option name as stored in the data sile
                        info["models"][f"{model_grp_name}/{model_opt_grp_name}"] = (
                            model_data
                        )
            return info

    def get_configs(self) -> list[EmbeddingConfig]:
        """Embedding configurations (model + preprocessing options) with
        features stored in this dataset, reconstructed from group attributes.
        Groups without config attributes or for unavailable models are skipped."""
        configs = []
        with h5py.File(self.storage_path, "r") as f:
            all_models_grp = f[self.schema.image_group].get(
                self.schema.all_models_group
            )
            if all_models_grp is None:
                return configs
            for model_grp in all_models_grp.values():
                for opt_grp in model_grp.values():
                    if self.schema.features_ds not in opt_grp:
                        continue
                    model = AVAILABLE_MODELS.get(
                        opt_grp.attrs.get(self.schema.attr_model)
                    )
                    if model is None:
                        continue
                    opts = {
                        opt: opt_grp.attrs[opt].item()
                        if hasattr(opt_grp.attrs[opt], "item")
                        else opt_grp.attrs[opt]
                        for opt in EmbeddingConfig.option_names()
                        if opt in opt_grp.attrs
                    }
                    configs.append(EmbeddingConfig(model, **opts))
        return configs

    def get_image_data(self) -> pl.DataFrame:
        return self.load_data()  # images paths only by default

    def load_data(
        self,
        config: EmbeddingConfig | None = None,
        paths=True,
        features=False,
        umap=False,
        clusters=False,
    ) -> pl.DataFrame:
        # if umap is requested, open in read/write mode in case we need to save
        read_mode = "r+" if umap else "r"
        with h5py.File(self.storage_path, read_mode) as f:
            img_grp = f[self.schema.image_group]
            # if umap is requested but has not yet been generated, calculate and save
            data = {}
            if paths:
                img_dataset = img_grp[self.schema.image_ds_paths]
                img_sizes = img_grp[self.schema.image_ds_sizes]
                # stored paths are relative to image dir; return full paths
                image_dir = Path(img_grp.attrs["image_dir"])
                data["image_path"] = [
                    str(image_dir / p) for p in img_dataset[:].astype("T")
                ]
                # return image sizes as well
                data["image_size"] = img_sizes[:]

            if features or umap:
                # model configuration is required if features/umap requested
                if config is None:
                    raise ValueError(
                        "model configuration is required to load features or umap"
                    )

                model_grp = f[self.schema.model_group(config, absolute=True)]

                features_dataset = None
                if umap:
                    # umap is requested
                    if self.schema.umap_ds not in model_grp:
                        print("** umap not present, calculating")
                        # load feature dataset
                        features_dataset = model_grp[self.schema.features_ds]
                        umap_coords = reduce_features(features_dataset)
                        # save as dataset for reuse
                        print("saving umap coords")
                        model_grp.create_dataset(
                            self.schema.umap_ds, data=umap_coords, compression="gzip"
                        )

                    else:
                        print("*** loading saved umap coords")
                        umap_coords = model_grp[self.schema.umap_ds][:]

                    # add to data dictionary to include in returned dataframe
                    data["umap"] = umap_coords

                if features:
                    # load feature dataset if not already loaded for umap
                    if features_dataset is None:
                        features_dataset = model_grp[self.schema.features_ds]
                    # [:] = retrieve all scalar data
                    data["features"] = features_dataset[:]

                # include cluster labels when requested and present in the data
                if clusters and self.schema.cluster_ds in model_grp:
                    cluster_dataset = model_grp[self.schema.cluster_ds]
                    data["cluster"] = cluster_dataset[:]

                # TODO: update to support cluster results for multiple k
                # maybe cluster/kmeans_{k} so they can be managed individually
                # move cluster generation logic to dataset; support cluster/save on demand?
                # then load as cluster_k{n} in dataframe when requested

            df = pl.DataFrame(data=data)
            # when image size is present, calculate aspect ratio
            if "image_size" in df.columns:
                df = df.with_columns(
                    aspect_ratio=pl.col.image_size.arr.first().truediv(
                        pl.col.image_size.arr.last()
                    )
                )

            return df
