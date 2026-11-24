"""Official pretrained LXMERT backbone and QA head initialization."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
THIRD_PARTY_LXMERT = REPO_ROOT / "third_party" / "lxmert"
DEFAULT_BERT_CONFIG = PACKAGE_ROOT / "assets" / "bert_config.json"
DEFAULT_ANSWER_TABLE = PACKAGE_ROOT / "assets" / "all_ans.json"


def official_modules():
    """Import upstream LXMERT modeling and tokenization modules."""
    path = str(THIRD_PARTY_LXMERT)
    if path not in sys.path:
        sys.path.insert(0, path)
    from lxrt import modeling, tokenization

    return modeling, tokenization


def official_tokenizer(bert_model: str = "bert-base-uncased"):
    """Load the BERT WordPiece tokenizer used by LXMERT."""
    _, tokenization = official_modules()
    return tokenization.BertTokenizer.from_pretrained(bert_model, do_lower_case=True)


def official_bert_config(config_path: str | Path | None = None):
    """Load the standard bert-base-uncased architecture configuration."""
    modeling, _ = official_modules()
    path = Path(config_path) if config_path is not None else DEFAULT_BERT_CONFIG
    return modeling.BertConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_official_qa_head(head, checkpoint_path, answers, answer_table_path=None):
    """Map pretrained LXMERT answer_head weights to the target VQA/GQA answer vocabulary.

    Reproduces official ``load_lxmert_qa`` answer normalization and row mapping
    with explicit file paths. Unmatched answers are zero-initialized.
    """
    official_modules()
    from pretrain.qa_answer_table import AnswerTable

    table_path = Path(answer_table_path) if answer_table_path is not None else DEFAULT_ANSWER_TABLE
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    prefix = "answer_head.logit_fc."
    loaded = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    expected = set(head.state_dict())
    if set(loaded) != expected:
        raise ValueError("Pretraining QA head keys do not match the official head")
    if not isinstance(head[-1], nn.Linear) or head[-1].out_features != len(answers):
        raise ValueError("QA head size does not match the target answer vocabulary")

    with table_path.open(encoding="utf-8") as handle:
        table = json.load(handle)
    pretrained_answers = [entry["ans"] for entry in table]
    if len(set(pretrained_answers)) != len(pretrained_answers):
        raise ValueError("Duplicate entries in the official pretraining answer table")
    answer_to_row = {answer: row for row, answer in enumerate(pretrained_answers)}
    source_weight, source_bias = loaded["3.weight"], loaded["3.bias"]
    if source_weight.shape[0] != len(table) or source_bias.shape[0] != len(table):
        raise ValueError("Pretraining answer table and QA weight rows disagree")
    if source_weight.shape[1:] != head[-1].weight.shape[1:]:
        raise ValueError("Pretraining QA feature dimension mismatch")

    normalizer = AnswerTable.__new__(AnswerTable)
    weight = torch.zeros_like(head[-1].weight)
    bias = torch.zeros_like(head[-1].bias)
    matched, missing = [], []
    for row, answer in enumerate(answers):
        normalized = normalizer.convert_ans(answer)
        source_row = answer_to_row.get(normalized)
        if source_row is None:
            missing.append(row)
        else:
            weight[row] = source_weight[source_row]
            bias[row] = source_bias[source_row]
            matched.append(row)
    loaded["3.weight"], loaded["3.bias"] = weight, bias
    head.load_state_dict(loaded, strict=True)
    return {
        "qa_head_initialization": "official_pretrained_answer_mapping",
        "qa_checkpoint": str(checkpoint_path),
        "qa_answer_table": str(table_path),
        "qa_matched_answers": len(matched),
        "qa_unmatched_answers": len(missing),
    }


class OfficialLxmertAdapter(nn.Module):
    """Wrapper around the official 9-5-5 pretrained LXMERT backbone (including pooler)."""

    def __init__(self, checkpoint_path: str | Path, bert_config_path: str | Path | None = None):
        super().__init__()
        modeling, _ = official_modules()
        modeling.VISUAL_CONFIG.l_layers = 9
        modeling.VISUAL_CONFIG.x_layers = 5
        modeling.VISUAL_CONFIG.r_layers = 5
        self.config = official_bert_config(bert_config_path)
        self.model = modeling.LXRTFeatureExtraction(self.config, mode="lxr")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backbone = {
            k.replace("module.", ""): v
            for k, v in state.items()
            if k.replace("module.", "").startswith("bert.")
        }
        self.model.load_state_dict(backbone, strict=True)

    @property
    def pooler(self):
        return self.model.bert.pooler

    def forward(
        self,
        input_ids,
        attention_mask,
        visual_feats,
        visual_pos,
        visual_attention_mask=None,
    ):
        (language, vision), pooled = self.model(
            input_ids,
            token_type_ids=torch.zeros_like(input_ids),
            attention_mask=attention_mask,
            visual_feats=(visual_feats, visual_pos),
            visual_attention_mask=visual_attention_mask,
        )
        return SimpleNamespace(
            language_output=language,
            vision_output=vision,
            pooled_output=pooled,
        )


class OfficialBaselineVQA(nn.Module):
    """Plain LXMERT baseline with official pretrained 9-5-5 backbone, pooler, and QA head."""

    def __init__(self, config: dict, encoder=None, head=None):
        super().__init__()
        self.num_answers = int(config["num_answers"])
        self.config = config
        self.lxmert = (
            encoder
            if encoder is not None
            else OfficialLxmertAdapter(
                config["official_checkpoint"],
                bert_config_path=config.get("bert_config_path"),
            )
        )
        self.initialization_metadata = {
            "backbone_initialization": (
                "injected"
                if encoder is not None
                else "official_pretrained_lxmert_9_5_5_including_pooler"
            ),
            "backbone_checkpoint": None if encoder is not None else str(config["official_checkpoint"]),
            "uses_pooler": True,
        }
        if head is None:
            modeling, _ = official_modules()
            hidden = int(self.lxmert.config.hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                modeling.GeLU(),
                modeling.BertLayerNorm(hidden * 2, eps=1e-12),
                nn.Linear(hidden * 2, self.num_answers),
            )
            self.head.apply(self.lxmert.model.init_bert_weights)
            with Path(config["answer_vocab_path"]).open(encoding="utf-8") as handle:
                vocabulary = json.load(handle)
            answers = (
                vocabulary
                if isinstance(vocabulary, list)
                else [vocabulary[str(row)] for row in range(len(vocabulary))]
            )
            if len(answers) != self.num_answers:
                raise ValueError("Answer vocabulary length mismatch")
            self.initialization_metadata.update(
                load_official_qa_head(
                    self.head,
                    config["official_checkpoint"],
                    answers,
                    config.get("pretrained_answer_table", DEFAULT_ANSWER_TABLE),
                )
            )
        else:
            self.head = head
            self.initialization_metadata["qa_head_initialization"] = "injected"

        self._baseline_parameter_count = sum(p.numel() for p in self.parameters())

    def parameter_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "baseline": self._baseline_parameter_count,
            "added": total - self._baseline_parameter_count,
            "ratio_to_baseline": total / self._baseline_parameter_count,
        }

    def training_diagnostics(self) -> dict:
        return {}

    def forward(
        self,
        visual_features,
        spatials,
        question_token_ids,
        question_token_mask,
    ):
        encoded = self.lxmert(
            input_ids=question_token_ids,
            attention_mask=question_token_mask,
            visual_feats=visual_features,
            visual_pos=spatials[..., :4],
        )
        scores = self.head(encoded.pooled_output)
        return {"ranking_scores": scores, "base_scores": scores, "diagnostics": {}}
