# SPDX-License-Identifier: Apache-2.0
#
# Ascend NPU forward for KimiGatedDeltaNetAttention (KDA).
#
# Upstream ``KimiGatedDeltaNetAttention._forward`` dispatches to vendor
# Triton kernels under ``vllm/models/kimi_k3/{amd,nvidia}/ops`` only; this
# subclass swaps in the Ascend Triton KDA kernels shipped with vllm-ascend
# (``vllm_ascend.ops.triton.kda``), which implement the same FLA-style
# KDA math (chunk prefill / fused recurrent decode).
#
# Everything else (weight layout, state shapes, conv1d, o_norm) is
# inherited unchanged from the upstream layer.

import torch
import os
from einops import rearrange

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm_ascend.ops.triton.kda.kda import chunk_kda, fused_recurrent_kda


def _gather_initial_states_npu(state, indices, has_initial_state):
    """NPU gather of dense state rows; upstream CUDA/PDL Triton kernel does
    not run on NPU, so use plain torch indexing + mask instead."""
    assert state.ndim >= 2
    assert indices.ndim == 1 and has_initial_state.ndim == 1
    assert indices.shape == has_initial_state.shape
    assert indices.device == state.device and has_initial_state.device == state.device
    assert indices.dtype in (torch.int32, torch.int64)
    assert has_initial_state.dtype == torch.bool

    output = torch.zeros(
        (indices.numel(), *state.shape[1:]),
        dtype=state.dtype,
        device=state.device,
    )
    mask = has_initial_state
    if bool(mask.any()):
        output[mask] = state[indices[mask]]
    return output


class AscendKimiGatedDeltaNetAttention(KimiGatedDeltaNetAttention):
    """KDA layer with the Ascend Triton kernel path."""

    def forward(self, hidden_states, positions, output=None):
        if output is None:
            output = torch.empty_like(hidden_states)
        super().forward(hidden_states, positions, output)
        return output

    def _forward(self, mixed_qkv, g1, g2, beta, core_attn_out):
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        m = attn_metadata_narrowed
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        num_actual_tokens = m.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]
        beta = torch.sigmoid(beta)
        if os.environ.get("KDA_DEBUG") == "1":
            print(f"[KDA] beta shape={tuple(beta.shape)} min={beta.min().item():.4f} max={beta.max().item():.4f} mean={beta.mean().item():.4f}", flush=True)
            print(f"[KDA] g1 shape={tuple(g1.shape)} min={g1.min().item():.4f} max={g1.max().item():.4f}", flush=True)
            print(f"[KDA] mixed_qkv shape={tuple(mixed_qkv.shape)} min={mixed_qkv.min().item():.4f} max={mixed_qkv.max().item():.4f}", flush=True)

        # Speculative decoding is not wired for v1: GLM-5.3 runs without a
        # drafter, so only the plain prefill / decode paths are supported.
        assert m.spec_sequence_masks is None, (
            "Ascend KDA path does not support spec-decode batches yet"
        )

        constant_caches = self.kv_cache
        conv_state, recurrent_state = constant_caches
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_state.split(
            self.local_projection_size, dim=-2
        )

        # Forget gate: raw f_b(f_a) -> per-channel log-decay g [1, T, H, D].
        # GLM/Kimi KDA use the bounded gate
        #   g = gate_lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))
        # (per-head A_log, per-channel dt_bias).  The upstream fused_kda_gate
        # instead implements the softplus form -exp(A_log)*softplus(raw_g+dt_bias),
        # which is only correct when no lower bound is configured.
        H = self.A_log.numel()
        D = self.head_dim
        raw_g = g1.reshape(-1, H * D).float().view(-1, H, D)
        bias = self.dt_bias.float().view(H, D)
        b_a = torch.exp(self.A_log.float()).view(H, 1)
        gate = self.gate_lower_bound * torch.sigmoid(b_a * (raw_g + bias.unsqueeze(0)))
        gate = gate.unsqueeze(0)

        if m.num_prefills > 0:
            q_ns, k_ns, v_ns = mixed_qkv.split(self.local_projection_size, dim=-1)

            def _prefill_conv(x, state, weight):
                return causal_conv1d_fn(
                    x.transpose(0, 1),
                    weight,
                    None,
                    activation="silu",
                    conv_states=state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=m,
                ).transpose(0, 1)

            q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
            k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
            v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
            q_ns, k_ns, v_ns = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in (q_ns, k_ns, v_ns)
            )

            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = _gather_initial_states_npu(
                recurrent_state,
                non_spec_state_indices_tensor,
                has_initial_state,
            )
            core_out, last_state = chunk_kda(
                q=q_ns,
                k=k_ns,
                v=v_ns,
                g=gate,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
            )
            recurrent_state[non_spec_state_indices_tensor] = last_state
            core_attn_out[0, :num_actual_tokens] = core_out
        else:
            decode_conv_indices = non_spec_state_indices_tensor[
                : mixed_qkv.size(0)
            ]
            packed_conv_out = torch.empty(
                mixed_qkv.shape,
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
                out=packed_conv_out,
            )
            q_ns, k_ns, v_ns = mixed_qkv.split(self.local_projection_size, dim=-1)
            q_ns, k_ns, v_ns = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in (q_ns, k_ns, v_ns)
            )
            core_out, _ = fused_recurrent_kda(
                q=q_ns,
                k=k_ns,
                v=v_ns,
                g=gate,
                beta=beta,
                initial_state=recurrent_state,
                ssm_state_indices=decode_conv_indices,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out[0, :num_actual_tokens] = core_out

        core_attn_out.copy_(self.o_norm(core_attn_out, g2))
