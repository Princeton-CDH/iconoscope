from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
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
