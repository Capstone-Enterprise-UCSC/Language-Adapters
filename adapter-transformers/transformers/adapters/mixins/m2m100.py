from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersWrapperMixin,
    ModelAdaptersMixin,
    ModelWithHeadsAdaptersMixin,
)


class M2M100EncoderLayerAdaptersMixin:
    """Adds adapters to the M2M100EncoderLayer module of M2M100."""

    def _init_adapter_modules(self):
        if not self.config.is_decoder:
            self.monolingual_enc_adapters = AdapterLayer("monolingual_enc_adapter", self.config)
            self.monolingual_enc_adapters._init_adapter_modules()
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class M2M100DecoderLayerAdaptersMixin(M2M100EncoderLayerAdaptersMixin):
    """Adds adapters to the M2M100DecoderLayer module of M2M100."""

    def _init_adapter_modules(self):
        super()._init_adapter_modules()
        self.cross_attention_adapters = AdapterLayer("cross_adapter", self.config)
        self.monolingual_dec_adapters = AdapterLayer("monolingual_dec_adapter", self.config)
        self.monolingual_dec_adapters._init_adapter_modules()
        self.cross_attention_adapters._init_adapter_modules()


class M2M100ModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelAdaptersMixin):
    """Adds adapters to the M2M100Model class."""

    invertible_adapters_base_name = "encoder"

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        if hasattr(self, "encoder"):
            for i, layer in enumerate(self.encoder.layers):
                yield i, layer
            for i, layer in enumerate(self.decoder.layers, start=len(self.encoder.layers)):
                yield i, layer
        else:
            for i, layer in enumerate(self.decoder.layers):
                yield i, layer


class M2M100ModelWithHeadsAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelWithHeadsAdaptersMixin):
    pass