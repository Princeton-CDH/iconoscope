from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class EmbedSpec:
    hf_id: str
    dim: int
    processor_kwargs: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({})
    )


EMBEDDINGS: dict[str, EmbedSpec] = {
    "dinov2": EmbedSpec("facebook/dinov2-base", 768),
}
DEFAULT_EMBEDDING = "dinov2"
