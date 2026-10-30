"""Generic representation-adaptive decoding for c-VEP BCIs."""

from .burst import (
    BayesianTargetAccumulator,
    BurstEvidenceConfig,
    BurstEvidenceDecoder,
    TargetEvidenceState,
)
from .datasets import (
    DEHAIS_OCCIPITAL_8,
    NBRS_OCCIPITAL_8,
    convert_dehais2024_subject,
    convert_nbrs_subject,
    convert_thielen2021_subject,
    load_nbrs_subject,
)
from .fbcca import NarrowBandFBCCA, NarrowBandFBCCAConfig
from .generic import GenericCvepDecoder, GenericDecoderState
from .grasp import GRASPDecoder
from .idle import BayesianIdleGate, IdleGateConfig

__all__ = [
    "BayesianIdleGate",
    "BayesianTargetAccumulator",
    "BurstEvidenceConfig",
    "BurstEvidenceDecoder",
    "DEHAIS_OCCIPITAL_8",
    "GenericCvepDecoder",
    "GenericDecoderState",
    "GRASPDecoder",
    "IdleGateConfig",
    "NarrowBandFBCCA",
    "NarrowBandFBCCAConfig",
    "NBRS_OCCIPITAL_8",
    "TargetEvidenceState",
    "convert_dehais2024_subject",
    "convert_nbrs_subject",
    "convert_thielen2021_subject",
    "load_nbrs_subject",
]

__version__ = "4.0.0"
