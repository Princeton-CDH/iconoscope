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
        "clip", "openai/clip-vit-base-patch32", 768, hf_model_class="CLIPVisionModel"
    ),
]

AVAILABLE_MODELS: dict[str, EmbeddingModel] = {model.name: model for model in _models}
DEFAULT_MODEL = _models[0]
