"""RG-LXMERT / ACM-LXMERT: LXMERT extended with projected CLIP RN50x4 grid tokens.

The 36 Faster R-CNN detector regions enter through the pretrained LXMERT visual
embedding. The CLIP grid cells of the same image are mapped by a trainable linear
projection + LayerNorm into the 768-d token space, combined with normalized cell
boxes via the pretrained box encoder and a zero-initialized type embedding, and
concatenated to the visual token sequence.

During fine-tuning under Asymmetric Configuration Mixing (ACM), per-sample
``use_regions`` flags mask all 36 region tokens in visual self-attention and
cross-modal attention (CLIP-only configuration) with probability ``p``, while
retaining both region and CLIP grid tokens (Joint configuration) with probability
``1 - p``. Standard inference uses the unmasked Joint configuration.
"""

from __future__ import annotations

import torch
from torch import nn

from acm_lxmert.models.lxmert_backbone import OfficialBaselineVQA, official_modules

CLIP_DIM = 2560
STORED_GRID = 7


def grid_boxes(grid: int) -> torch.Tensor:
    """Normalized boxes (x1, y1, x2, y2) for a grid x grid layout in row-major order."""
    edges = torch.linspace(0.0, 1.0, grid + 1)
    rows, columns = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return torch.stack(
        [edges[columns], edges[rows], edges[columns + 1], edges[rows + 1]],
        dim=-1,
    ).reshape(grid * grid, 4)


class ClipGridVQA(OfficialBaselineVQA):
    """Joint Region--CLIP Grid architecture shared by RG-LXMERT and ACM-LXMERT."""

    def __init__(self, config: dict, grid: int = STORED_GRID, encoder=None, head=None):
        super().__init__(config, encoder=encoder, head=head)
        if not 1 <= grid <= STORED_GRID:
            raise ValueError(f"grid must be between 1 and {STORED_GRID}")
        self.grid = int(grid)
        hidden = int(self.lxmert.config.hidden_size)
        modeling, _ = official_modules()
        with torch.random.fork_rng(devices=[]):
            self.clip_fc = nn.Linear(CLIP_DIM, hidden)
            self.clip_layer_norm = modeling.BertLayerNorm(hidden, eps=1e-12)
            self.clip_fc.apply(self.lxmert.model.init_bert_weights)
            self.clip_layer_norm.apply(self.lxmert.model.init_bert_weights)
        self.clip_type = nn.Parameter(torch.zeros(hidden))
        self.register_buffer("clip_boxes", grid_boxes(self.grid), persistent=False)
        self.initialization_metadata.update(
            method="clip_grid_tokens_appended_to_the_visual_sequence",
            clip_model="CLIP RN50x4, 448 input",
            cells=self.grid * self.grid,
            new_parameters=sum(
                p.numel()
                for p in (
                    *self.clip_fc.parameters(),
                    *self.clip_layer_norm.parameters(),
                    self.clip_type,
                )
            ),
        )

    def clip_tokens(self, clip_grid: torch.Tensor) -> torch.Tensor:
        """Project stored [B, 49, 2560] CLIP cells to [B, grid*grid, hidden] tokens."""
        x = clip_grid.float()
        if self.grid != STORED_GRID:
            batch = x.size(0)
            x = x.transpose(1, 2).reshape(batch, CLIP_DIM, STORED_GRID, STORED_GRID)
            x = nn.functional.adaptive_avg_pool2d(x, self.grid).flatten(2).transpose(1, 2)
        encoder = self.lxmert.model.bert.encoder.visn_fc
        x = self.clip_layer_norm(self.clip_fc(x))
        y = encoder.box_layer_norm(encoder.box_fc(self.clip_boxes.to(x.dtype)))
        return encoder.dropout((x + y[None]) / 2 + self.clip_type)

    def forward(
        self,
        visual_features,
        spatials,
        question_token_ids,
        question_token_mask,
        clip_grid=None,
        use_clip=None,
        use_regions=None,
    ):
        bert = self.lxmert.model.bert
        encoder = bert.encoder
        language = bert.embeddings(question_token_ids, torch.zeros_like(question_token_ids))
        language_mask = (1.0 - question_token_mask[:, None, None, :].to(language.dtype)) * -10000.0
        vision = encoder.visn_fc((visual_features, spatials[..., :4]))
        vision_mask = None
        if clip_grid is not None:
            extra = self.clip_tokens(clip_grid)
            regions = vision.size(1)
            vision = torch.cat([vision, extra], dim=1)
            if use_clip is not None or use_regions is not None:
                vision_mask = torch.zeros(
                    vision.size(0),
                    1,
                    1,
                    vision.size(1),
                    dtype=vision.dtype,
                    device=vision.device,
                )
                if use_clip is not None:
                    vision_mask[:, :, :, regions:] = (
                        (1.0 - use_clip.to(vision.dtype))[:, None, None, None] * -10000.0
                    )
                if use_regions is not None:
                    vision_mask[:, :, :, :regions] = (
                        (1.0 - use_regions.to(vision.dtype))[:, None, None, None] * -10000.0
                    )
        for layer in encoder.layer:
            language = layer(language, language_mask)
        for layer in encoder.r_layers:
            vision = layer(vision, vision_mask)
        for layer in encoder.x_layers:
            language, vision = layer.cross_att(language, language_mask, vision, vision_mask)
            language = layer.lang_self_att(language, language_mask)
            vision = layer.visn_self_att(vision, vision_mask)
            language, vision = layer.output_fc(language, vision)
        scores = self.head(bert.pooler(language))
        return {"ranking_scores": scores, "base_scores": scores, "diagnostics": {}}

    def training_diagnostics(self) -> dict:
        return {
            "clip_fc_weight_norm": float(self.clip_fc.weight.detach().norm()),
            "clip_type_norm": float(self.clip_type.detach().norm()),
        }
