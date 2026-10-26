# Changelog

## [0.2] 2026-10-26

Complete rewrite of the package; uses `transformers` python library to use a consistent interface for loading and running models. HDF5 files from 0.1 are _not compatible_ with v0.2.

Provides a command-line tool to work with image embeddings in a persistent, flexible image dataset stored in a custom HDF5 file. Includes the following functionality:

- **Create** an image dataset based on a directory of images; saves as an HDF5 file with image paths
  - Uses `PIL` to verify images before adding to dataset; stores base directory and relative paths for portability.
- **Embed** images with one or more vision models with optional pre-processing and save features in the image dataset file
  - Supports DINOv2, DINOv3, ResNet-50, CLIP, SigLIP
  - Supports center-cropping by % of image and grayscale
  - HDF5 support saving features from multiple models and configurations in parallel.
- **Mosaic** : generate a static mosaic from dataset images (or a subset); runs PCA and UMAP embeddings and then slots images into a grid
  - Uses UMAP coordinates for specified model and preprocessing options; generates and saves UMAP if not already present.
- **UMAP**: programmatically generate and save UMAP coordinates for any or all model features present in an image dataset file
- **Cluster**: Run k-means with a specified K on a set of embeddings and save the results to the HDF5 file
- **Info**: Report the contents of an iconoscope image dataset (number of image paths, extracted features, and any related UMAP coordinates or clustering)

An image dataset created by iconoscope v0.2 includes a schema and version number, with light validation of required fields.

### Notebooks

The repository includes experimental Marimo notebooks for exploring image datasets:

- UMAP → grid animation, to explain how the image mosaic layout works
- Nearest Neighbor search with comparison across models
- Princeton Geniza Project image embedding explorer; autoloads a prepared image dataset, and makes available for both UMAP → grid and nearest neighbor search

## [0.1.1] 2026-06-18

- Make CLIP installation optional and manual, since PyPI doesn't allow it as a dependency.

## [0.1] 2026-06-18

Initial prototype; started as a vibe-coded prototype which started as a python port of [Andrej Karpathy's CNN embedding visualizer](https://cs.stanford.edu/people/karpathy/cnnembed/) created with ClaudeCode.

Provides a command line tool to embed a directory of images using pretrained vision models and generate static images of grid mosaics based on UMAP or t-SNE coordinates from the embeddings.

- Embed: Extract features from a directory of images and save to an HDF5 file. Supports DINOv2 (default) and CLIP.
- Mosaic: Reduce embeddings to 2D and render a static mosaic where images are slotted into grid based on 2D layout. 2D coordinates are cached in the HDF5 file.
- Viewer: Interactive HTML viewer as a pre-built Svelte app.
- Cluster: Run k-means clustering on saved embedding
- Info: Report what is stored in an embeddings file

[0.1]: https://github.com/Princeton-CDH/iconoscope/releases/tag/0.1
[0.1.1]: https://github.com/Princeton-CDH/iconoscope/releases/tag/0.1.1
[0.2]: https://github.com/Princeton-CDH/iconoscope/releases/tag/0.2
