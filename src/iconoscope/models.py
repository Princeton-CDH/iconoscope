from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    #: shorthand name
    name: str
    #: pretrained model name, for use with hugging face from_pretrained method
    id: str
    #: dimensions of the model's output feature vector
    dimensions: int


_models = [EmbeddingModel("dinov2", "facebook/dinov2-base", 768)]

AVAILABLE_MODELS: dict[str, EmbeddingModel] = {model.name: model for model in _models}
DEFAULT_MODEL = _models[0]
