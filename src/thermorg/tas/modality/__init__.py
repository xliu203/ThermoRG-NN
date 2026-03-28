# SPDX-License-Identifier: Apache-2.0

"""Data modality support for TAS.

This module provides support for different data modalities (tabular, text, video, audio)
through embeddings and specialized distance metrics.
"""

from .base import BaseModality, ModalityConfig
from .tabular import TabularModality
from .embedding import EmbeddingModality
from .registry import ModalityRegistry

__all__ = [
    'BaseModality',
    'ModalityConfig',
    'TabularModality',
    'EmbeddingModality',
    'ModalityRegistry',
]
