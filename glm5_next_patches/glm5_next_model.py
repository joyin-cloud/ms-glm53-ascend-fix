# SPDX-License-Identifier: Apache-2.0
#
# Inference-only GLM-5.3-Flash (``Glm5NextForConditionalGeneration``) text
# model for Ascend NPU.
#
# 320B total / 18B active. 45 text layers in a hybrid pattern: 34 KDA
# linear-attention layers + 11 NoPE-MLA (DSA) layers, 4-stream
# manifold-constrained hyper-connections (mHC) at every attention / FFN
# site, 288 routed experts (sigmoid, e_score_correction_bias) + 1 shared
# expert, block-FP8 weights.
#
# v1 scope (see /root/glm53-port/ref/glm53_recipe.md):
#   * text path only -- the checkpoint's ``model.visual.*`` tower is skipped
#     at load; the MTP draft layer (checkpoint layer 45) is dropped.
#   * DSA layers run dense. With ``index_topk=2048`` / ``index_kpool=4`` the
#     indexer selects everything for sequences <= 2048 tokens, so dense
#     causal MLA is numerically identical there. Beyond that the sparse
#     k-pool path is not wired yet.
#   * KDA forward uses the Ascend Triton KDA kernels (kda_ascend.py).
#
# Reference: ROCm ATOM's ``atom/models/glm5_next.py`` (same architecture,
# GPU kernels) and vllm-ascend's DeepSeek-V4 model (mHC + NPU MoE wiring).

import typing
import os
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_ascend.models.deepseek_v4.model import DeepseekV2MLP, DeepseekV4MoE

from .kda_ascend import AscendKimiGatedDeltaNetAttention


def _build_kimi_linear_config(text) -> KimiLinearConfig:
    """Wrap the raw GLM-5.3 text config as a KimiLinearConfig.

    Kimi's ``is_kda_layer(idx)`` is 1-based (``(idx + 1) in kda_layers``)
    while the GLM checkpoint lists layers 0-based -- shift both lists by +1.
    """
    lin = text.linear_attn_config
    kda = [int(i) + 1 for i in lin["kda_layers"]]
    full = [int(i) + 1 for i in lin["full_attn_layers"]]
    shifted = dict(lin)
    shifted["kda_layers"] = kda
    shifted["full_attn_layers"] = full

    cfg = KimiLinearConfig(
        vocab_size=text.vocab_size,
        hidden_size=text.hidden_size,
        num_hidden_layers=text.num_hidden_layers,
        num_attention_heads=text.num_attention_heads,
        rms_norm_eps=text.rms_norm_eps,
        hidden_act="silu",
        q_lora_rank=text.q_lora_rank,
        kv_lora_rank=text.kv_lora_rank,
        qk_nope_head_dim=text.qk_nope_head_dim,
        qk_rope_head_dim=text.qk_rope_head_dim,
        v_head_dim=text.v_head_dim,
        mla_use_nope=True,
        num_experts=text.n_routed_experts,
        num_experts_per_token=text.num_experts_per_tok,
        num_shared_experts=text.n_shared_experts,
        moe_router_activation_func="sigmoid",
        moe_renormalize=bool(text.norm_topk_prob),
        routed_scaling_factor=text.routed_scaling_factor,
        moe_intermediate_size=text.moe_intermediate_size,
        intermediate_size=text.intermediate_size,
        first_k_dense_replace=text.first_k_dense_replace,
        moe_layer_freq=1,
        linear_attn_config=shifted,
        max_position_embeddings=text.max_position_embeddings,
        num_nextn_predict_layers=text.num_nextn_predict_layers,
        tie_word_embeddings=getattr(text, "tie_word_embeddings", False),
    )
    # Aliases read by the reused DeepSeek-V4 MoE / MLP classes.
    cfg.n_routed_experts = text.n_routed_experts
    cfg.n_shared_experts = text.n_shared_experts
    cfg.norm_topk_prob = text.norm_topk_prob
    cfg.num_experts_per_tok = text.num_experts_per_tok
    cfg.scoring_func = "sigmoid"
    cfg.swiglu_limit = text.swiglu_limit
    cfg.num_hash_layers = 0
    # mHC hyper-connection parameters (read by the decoder layer).
    cfg.hc_mult = text.hc_mult
    cfg.hc_sinkhorn_iters = text.hc_sinkhorn_iters
    cfg.hc_eps = text.hc_eps
    return cfg


class Glm5NextDecoderLayer(nn.Module):
    """One hybrid layer: (mHC -> KDA or dense-MLA) then (mHC -> MoE/dense)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        cfg: KimiLinearConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = cfg
        self.layer_idx = int(prefix.rsplit(".", 1)[-1])
        quant_config = vllm_config.quant_config

        if cfg.is_kda_layer(self.layer_idx):
            self.self_attn = AscendKimiGatedDeltaNetAttention(
                cfg, vllm_config, prefix=f"{prefix}.self_attn"
            )
            self.is_linear_attn = True
        else:
            self.is_linear_attn = False
            self.self_attn = DeepseekV2MLAAttention(
                vllm_config,
                config=cfg,
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                q_lora_rank=cfg.q_lora_rank,
                kv_lora_rank=cfg.kv_lora_rank,
                max_position_embeddings=cfg.max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        if self.layer_idx >= cfg.first_k_dense_replace:
            self.mlp = DeepseekV4MoE(
                config=cfg,
                parallel_config=vllm_config.parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                hidden_act=cfg.hidden_act,
                swiglu_limit=cfg.swiglu_limit,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # mHC parameters are flat on the layer in the checkpoint
        # (``layers.N.hc_attn_fn`` / ``hc_attn_base`` / ``hc_attn_scale``),
        # all fp32.
        hc_mult = cfg.hc_mult
        mix = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * cfg.hidden_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.hc_attn_fn = nn.Parameter(torch.empty(mix, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x, hc_fn, hc_scale, hc_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps,
        )

    def hc_post(self, x, residual, post, comb):
        y = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0), residual.unsqueeze(dim=0),
            post.unsqueeze(dim=0), comb.unsqueeze(dim=0),
        )
        return y.squeeze(dim=0)

    def forward(self, positions, hidden_states, residual=None):
        # ``hidden_states`` is the widened mHC residual stream [T, hc, D].
        _is_prefill = hidden_states.shape[0] > 1
        _g2 = os.environ.get("GLM53_DUMP2") == "1" and get_tensor_model_parallel_rank() == 0 and _is_prefill
        _save2 = lambda name, x: torch.save(x.float().cpu(), f"/data/dump2/L{self.layer_idx}.{name}.pt") if _g2 and x is not None else None
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        _save2("attn_hc_collapsed", hidden_states); _save2("attn_hc_post", post); _save2("attn_hc_comb", comb)
        hidden_states = self.input_layernorm(hidden_states)
        _save2("iln", hidden_states)
        if self.is_linear_attn:
            hidden_states = self.self_attn(hidden_states, positions)
        else:
            hidden_states = self.self_attn(positions, hidden_states, None)
        _save2("attn", hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        _save2("attn_out", hidden_states)

        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        _save2("ffn_hc_collapsed", hidden_states); _save2("ffn_hc_post", post); _save2("ffn_hc_comb", comb)
        hidden_states = self.post_attention_layernorm(hidden_states)
        _save2("pln", hidden_states)
        hidden_states = self.mlp(hidden_states)
        _save2("mlp", hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        return hidden_states


class Glm5NextModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        cfg: KimiLinearConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = cfg
        self.hc_mult = cfg.hc_mult
        self.vocab_size = cfg.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                cfg.vocab_size, cfg.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            cfg.num_hidden_layers,
            lambda prefix: Glm5NextDecoderLayer(
                vllm_config, cfg, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds=None,
    ):
        if get_pp_group().is_first_rank:
            hidden = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            # Widen into the mHC residual stream: every stream starts as the
            # embedding.
            residual = hidden.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()
        else:
            assert intermediate_tensors is not None
            residual = intermediate_tensors["residual"]

        _dbg_rank = get_tensor_model_parallel_rank()
        _dump = os.environ.get("GLM53_DUMP") == "1" and _dbg_rank == 0
        _dump_gate = _dump and residual.shape[0] > 1
        if _dump_gate and get_pp_group().is_first_rank:
            torch.save(residual.float().cpu(), f"/data/dump/emb.pt")
            torch.save(input_ids.cpu(), "/data/dump/input_ids.pt")
            print(f"[DUMP] emb shape={tuple(residual.shape)} ids={input_ids.cpu().tolist()}", flush=True)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            residual = layer(positions, residual)
            if _dbg_rank == 0 and _dbg_active() and residual.shape[0] > 1:
                f = residual.float()
                n = torch.isnan(f).any().item()
                print(f"[DBG] layer {layer.layer_idx} mean={f.mean().item():.5g} absmax={f.abs().max().item():.5g} nan={n}", flush=True)
            if _dump_gate:
                torch.save(residual.float().cpu(), f"/data/dump/layer_{layer.layer_idx}.pt")
                print(f"[DUMP] layer {layer.layer_idx} shape={tuple(residual.shape)}", flush=True)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"residual": residual})

        # GLM collapses the hc streams with an unweighted mean right before
        # the final norm (no learned reduction).
        if _dbg_rank == 0 and _dbg_active() and residual.shape[0] > 1:
            fm = residual.float()
            print(f"[DBG] FINAL hc mean={fm.mean().item():.5g} absmax={fm.abs().max().item():.5g} nan={torch.isnan(fm).any().item()}", flush=True)
        _collapsed = residual.mean(dim=-2)
        if _dump_gate:
            torch.save(_collapsed.float().cpu(), "/data/dump/final_hidden.pt")
            print(f"[DUMP] final_hidden shape={tuple(_collapsed.shape)}", flush=True)
        return self.norm(_collapsed)



def _dbg_active() -> bool:
    """True only during real serving (not the profile dummy run, which runs
    without a forward context) -- safe for eager mode (no graph capture)."""
    from vllm.forward_context import is_forward_context_available
    return is_forward_context_available()


class AscendGlm5NextForConditionalGeneration(
    nn.Module, SupportsPP, MixtureOfExperts
):
    """GLM-5.3-Flash, text path (vision tower and MTP draft not served)."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    model_cls = Glm5NextModel

    moe_mlp_layers: list
    moe_layers: list

    is_hybrid = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )
        text = vllm_config.model_config.hf_config.text_config
        lin = text.linear_attn_config
        return MambaStateShapeCalculator.kda_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            num_heads=lin["num_heads"],
            head_dim=lin["head_dim"],
            conv_kernel_size=lin.get("short_conv_kernel_size", 4),
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.hf_config = hf_config
        text = hf_config.text_config
        self.text_config = text
        self.config = _build_kimi_linear_config(text)
        self.quant_config = vllm_config.quant_config

        self.model = self.model_cls(
            vllm_config=vllm_config,
            cfg=self.config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                text.vocab_size,
                text.hidden_size,
                quant_config=None,  # bf16 in the checkpoint
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(text.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_moe_layers = self.config.num_hidden_layers
        self.num_expert_groups = getattr(self.text_config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, DeepseekV4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def extract_moe_parameters(self, example_moe: DeepseekV4MoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ):
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds=None,
    ):
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if os.environ.get("GLM53_DUMP") == "1" and get_tensor_model_parallel_rank() == 0 and logits is not None and hidden_states.shape[0] > 1:
            torch.save(logits.float().cpu(), "/data/dump/logits.pt")
            print(f"[DUMP] logits shape={tuple(logits.shape)}", flush=True)
        if get_tensor_model_parallel_rank() == 0 and logits is not None and _dbg_active():
            f = logits.float()
            print(f"[DBG] logits mean={f.mean().item():.5g} absmax={f.abs().max().item():.5g} nan={torch.isnan(f).any().item()} finites={torch.isfinite(f).sum().item()}/{f.numel()}", flush=True)
        return logits

    def get_expert_mapping(self):
        from vllm_ascend.ascend_config import get_ascend_config

        mix_placement = getattr(get_ascend_config(), "mix_placement", False)
        return fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.text_config.n_routed_experts
            + (self.text_config.n_shared_experts if mix_placement else 0),
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Adapted verbatim in structure from vllm_ascend's DeepSeek-V4
        # loader (identical MoE layout):
        #   * ``model.language_model.`` -> ``model.`` checkpoint rename,
        #   * stacked params (dense gate_up, KDA in_proj/conv1d, MLA
        #     fused_qkv_a_proj) via per-layer weight loaders,
        #   * expert weights via the RoutedExperts expert-aware loaders,
        #   * the MTP draft layer (checkpoint layer 45), the unused vision
        #     tower, and the DSA indexer weights (v1 runs dense) skipped.
        mix_placement = getattr(get_ascend_config(), "mix_placement", False)

        stacked_params_mapping = [
            # MLA compressed front-end (q_lora | kv_lora) -- keep the most
            # specific match first.
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            # KDA packed in_proj (q | k | v | beta | f_a)
            ("in_proj_qkvgfab", "q_proj", 0),
            ("in_proj_qkvgfab", "k_proj", 1),
            ("in_proj_qkvgfab", "v_proj", 2),
            ("in_proj_qkvgfab", "b_proj", 3),
            ("in_proj_qkvgfab", "f_a_proj", 4),
            # KDA packed conv1d (q | k | v)
            ("conv1d", "q_conv1d", 0),
            ("conv1d", "k_conv1d", 1),
            ("conv1d", "v_conv1d", 2),
            # dense FFN / shared experts
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if mix_placement else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        for name, loaded_weight in weights:
            if not name.startswith("model"):
                name = f"model.{name}"
            # The checkpoint nests the LM under model.language_model.*
            name = name.replace("model.language_model.", "model.", 1)
            # KDA checkpoint nests forget-gate linears under self_attn.forget_gate.*
            name = name.replace(".self_attn.forget_gate.", ".self_attn.", 1)
            # mHC params are flat in the checkpoint (attn_hc/ffn_hc.{fn,base,scale})
            name = name.replace(".attn_hc.", ".hc_attn_", 1)
            name = name.replace(".ffn_hc.", ".hc_ffn_", 1)
            # The LM head lives at the top level of the module tree.
            if name.startswith("model.lm_head."):
                name = name.replace("model.lm_head.", "lm_head.", 1)

            # Vision tower: not served on the text path.
            if name.startswith("model.visual."):
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip the MTP draft layer

            # v1 runs the DSA layers dense; the indexer weights are unused.
            if ".indexer." in name:
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Expert weights are handled by the expert mapping below.
                if "mlp.experts." in name and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                name = name_mapped
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # Expert weight not owned by this rank: skip.
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    if name.endswith(".conv1d.weight"):
                        # single fused (3*proj, 1, k) conv weight in the
                        # checkpoint; the module's loader expects q/k/v calls
                        w = loaded_weight
                        if w.dim() == 2:
                            w = w.unsqueeze(1)
                        p_size = w.shape[0] // 3
                        wl = param.weight_loader
                        for sid in range(3):
                            wl(param, w[sid * p_size:(sid + 1) * p_size].contiguous(), sid)
                    else:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
