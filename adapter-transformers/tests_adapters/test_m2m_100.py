import unittest

from transformers import M2M100Config
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    UniPELTTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .composition.test_parallel import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class M2M100AdapterTestBase(AdapterTestBase):
    config_class = M2M100Config
    config = make_config(
        M2M100Config,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=250027,
    )
    tokenizer_name = "facebook/m2m100_418M"


@require_torch
class M2M100AdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    M2M100AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class M2M100ClassConversionTest(
    ModelClassConversionTestMixin,
    M2M100AdapterTestBase,
    unittest.TestCase,
):
    pass
