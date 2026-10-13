from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class EmbeddingModel:
    #: shorthand name
    name: str
    #: pretained model name, for use with hugging face from_pretained method
    id: str
    #: default dimensions for feature vector
    dimensions: int
    processor_kwargs: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )


_models = [EmbeddingModel("dinov2", "facebook/dinov2-base", 768)]

AVAILABLE_MODELS: dict[str, EmbeddingModel] = {model.name: model for model in _models}
DEFAULT_MODEL = _models[0]
