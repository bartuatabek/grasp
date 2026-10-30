"""Public GRASP decoder API.

This module is intentionally small: the implementation remains split into
representation, burst-evidence, FBCCA, and idle-gate components, while this
facade gives experiments and downstream users one stable import name.
"""

from .generic import GenericCvepDecoder, GenericDecoderState


class GRASPDecoder(GenericCvepDecoder):
    """Generic Representation-Adaptive Sequential Prediction decoder.

    The model is participant-specific and is fitted with EEG, sampling rate,
    codebook, and labels only.  It inherits the tested implementation and
    public methods of :class:`GenericCvepDecoder` (``fit``, ``update``,
    ``fit_idle``, ``set_codebook``, and ``feedback``).
    """


__all__ = ["GRASPDecoder", "GenericCvepDecoder", "GenericDecoderState"]
