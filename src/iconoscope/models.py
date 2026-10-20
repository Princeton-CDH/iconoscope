from dataclasses import dataclass, fields
from functools import cached_property


@dataclass(frozen=True)
class EmbeddingModel:
    """Configuration for supported vision or multimodal models."""

    #: shorthand name
    name: str
    #: pretrained model name, for use with hugging face from_pretrained method
    id: str
    #: dimensions of the model's output feature vector
    dimensions: int
    #: checkpoint key prefixes that are expected to be unused when loading
    #: the base model (e.g. classification heads); load warnings that
    #: *only* concern these keys are suppressed
    ignore_unexpected: tuple[str, ...] = ()
    #: optional transformers model class to use, for models that require something other than AutoModel
    hf_model_class: str | None = None


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for a specific model and pre-processing options."""

    model: EmbeddingModel
    center_crop: float | None = None
    grayscale: bool = False

    def __post_init__(self):
        if self.center_crop is not None:
            if not 0.0 < self.center_crop <= 1.0:
                raise ValueError(
                    f"center_crop must be a fraction between 0 and 1 (got {self.center_crop})"
                )

    @classmethod
    def option_names(cls) -> list[str]:
        "names of fields corresponding to supported pre-processing options"
        return [f.name for f in fields(cls) if f.name != "model"]

    @cached_property
    def opts(self) -> dict:
        return {f: getattr(self, f) for f in self.option_names()}

    @cached_property
    def option_labels(self) -> list[str]:
        """One short label per set option, e.g. ``["center_crop=0.6", "grayscale"]``."""
        opt_labels = []
        for key, value in self.opts.items():
            # if property is unset or False, omit
            if value is None or value is False:
                continue
            # if property is boolean True, include property name only
            if value is True:
                opt_labels.append(key)
            # otherwise, include both key and value
            else:
                # format floats compactly (0.6 not 0.6000000001)
                if isinstance(value, float):
                    value = f"{value:g}"
                opt_labels.append(f"{key}={value}")
        return opt_labels

    @cached_property
    def key(self) -> str:
        # if no customization, return "defaults"
        return "_".join(self.option_labels) or "defaults"

    @property
    def label(self) -> str:
        """Display label: ``model/key``."""
        return f"{self.model.name}/{self.key}"


_models = [
    EmbeddingModel("dinov2", "facebook/dinov2-base", 768),
    EmbeddingModel("dinov3", "facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
    # checkpoint is for image classification; we only use the base model
    EmbeddingModel(
        "resnet", "microsoft/resnet-50", 2048, ignore_unexpected=("classifier.",)
    ),
    EmbeddingModel(
        "siglip",
        "google/siglip-base-patch16-224",
        768,
        hf_model_class="SiglipVisionModel",
        ignore_unexpected=("text_model.", "logit_bias", "logit_scale"),
    ),
    # CLIP vision tower only; the checkpoint also contains the text tower and
    # projection layers, which are unused here.
    #
    # NOTE: features are the vision tower's 768-dim `pooler_output`, which is
    # NOT in the shared image-text embedding space. To support text-to-image
    # search in the future:
    #   - use hf_model_class="CLIPVisionModelWithProjection", dimensions=512,
    #     and read `image_embeds` instead of `pooler_output` (would need a
    #     configurable output attribute on EmbeddingModel; see embed.py)
    #   - drop "visual_projection." from ignore_unexpected, since it is used
    #   - embed queries with CLIPTextModelWithProjection + AutoTokenizer
    #     (`text_embeds`, 512-dim), L2-normalize both image and text
    #     embeddings, and rank images by dot product (cosine similarity)
    # Consider registering as a separate model (e.g. "clip-proj") so existing
    # datasets with 768-dim "clip" features remain valid.
    EmbeddingModel(
        "clip",
        "openai/clip-vit-base-patch32",
        768,
        hf_model_class="CLIPVisionModel",
        ignore_unexpected=(
            "text_model.",
            "text_projection.",
            "visual_projection.",
            "logit_scale",
        ),
    ),
]

AVAILABLE_MODELS: dict[str, EmbeddingModel] = {model.name: model for model in _models}
DEFAULT_MODEL = _models[0]
