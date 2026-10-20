"""
VisDINO Modeling Modules

Core innovation: Occlusion-Gated Sampling (OGS) in deformable cross-attention,
supervised by CrowdHuman visible/full box annotations.

Modules:
- ogs_attention: OGS-enhanced MultiScaleDeformableAttention
- visdino_transformer: VisDINO encoder-decoder with OGS + dual-box refinement
- visdino: VisDINO model (extends DINO with visibility awareness)
- visdino_criterion: Multi-task loss with L_point, L_vbox, L_ratio
"""

from .ogs_attention import OGSMultiScaleDeformableAttention
from .visdino_transformer import (
    VisDINOTransformerDecoder,
    VisDINOTransformer,
)
from .visdino import VisDINO
from .visdino_criterion import VisDINOCriterion

__all__ = [
    "OGSMultiScaleDeformableAttention",
    "VisDINOTransformerDecoder",
    "VisDINOTransformer",
    "VisDINO",
    "VisDINOCriterion",
    "VisDINOSetCriterion",
]
