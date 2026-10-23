import argparse

from iconoscope.commands import load_dataset


def _dimensions(shape: tuple[int, int]) -> str:
    return "x".join(str(dim) for dim in shape)


def main(args: argparse.Namespace) -> None:
    img_dataset = load_dataset(args.dataset)
    details = img_dataset.info()
    info_details = [f"Details for {args.dataset} :"]

    if "schema" in details:
        info_details.append(f"  schema: {details['schema']}")

    if "image_paths" in details:
        whence = f" from {details['image_dir']}" if details.get("image_dir") else ""
        info_details.append(f"  {details['image_paths']:,} image paths{whence}")
        if "image_size_mode" in details:
            info_details.append(
                f"\tMost frequent image size:  {'x'.join(str(dim) for dim in details['image_size_mode'])}"
            )
        if "image_aspect_mode" in details:
            # todo: might be nice display in width:height format also, based on image mode and gcd
            # round to nearest ten with round(n, -1); then use math.gcd to reduce to lowest terms for width:height display
            info_details.append(
                f"\tMost frequent aspect ratio: {details['image_aspect_mode']:.2f}"
            )
    else:
        info_details.append(" image_path and features columns not found")

    for model, model_details in details["models"].items():
        embed_dims = _dimensions(model_details["embeddings"])
        info_details.append(f"  {model} extracted features ({embed_dims})")
        if "umap" in model_details:
            info_details.append(
                f"\tumap coordinates ({_dimensions(model_details['umap'])})"
            )
        for cluster in model_details.get("clusters", []):
            info_details.append(
                f"\t{cluster['size']:,} cluster labels "
                f"({cluster['method']}, k={cluster['n_clusters']})"
            )

    print("\n".join(info_details))
