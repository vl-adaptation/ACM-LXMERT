"""ACM-LXMERT: Asymmetric Configuration Mixing for CLIP Grid Feature Adaptation in VQA."""

from acm_lxmert.models.acm_model import ClipGridVQA, grid_boxes
from acm_lxmert.models.lxmert_backbone import OfficialBaselineVQA, OfficialLxmertAdapter

__all__ = [
    "ClipGridVQA",
    "OfficialBaselineVQA",
    "OfficialLxmertAdapter",
    "grid_boxes",
]
