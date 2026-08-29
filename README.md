# GLM-5.3-Flash on Ascend 910B — vLLM Fix

## Problem
GLM-5.3-Flash served via vLLM-Ascend on Huawei 910B 8×NPU produced garbage output.
The root cause was a missing `sigmoid` activation on the `beta` parameter in the
KDA (Kimi Gated Delta Attention) forward path.

## Root Cause
In the reference implementation (transformers `Glm5NextTextLinearAttention`):
```python
beta = torch.sigmoid(self.b_proj(hidden_states))
```

In vLLM-Ascend's `AscendKimiGatedDeltaNetAttention._forward`, `beta` was used
directly from the `in_proj_qkvgfab` projection output **without** sigmoid. This
caused beta values to range from -1 to +2.4 instead of [0, 1], corrupting the
KDA attention's key/value scaling (`k_beta = key * beta`, `v_beta = value * beta`).

## Fix
Add `beta = torch.sigmoid(beta)` in `kda_ascend.py`'s `_forward` method, after
the beta tensor is sliced to `num_actual_tokens`.

## Files
- `glm5_next_patches/kda_ascend.py` — patched KDA attention with sigmoid fix
- `glm5_next_patches/glm5_next_model.py` — GLM-5.3-Flash model for vLLM-Ascend

## Deployment
Copy patches into the vLLM-Ascend container and restart vLLM serve:
```bash
cp /data/patch/kda_ascend.py /vllm-workspace/vllm-ascend/vllm_ascend/models/glm5_next/kda_ascend.py
cp /data/patch/glm5_next_model.py /vllm-workspace/vllm-ascend/vllm_ascend/models/glm5_next/model.py
```

## Performance (8×910B, w8a8, TP=8, eager mode)
| Concurrency | Throughput (tok/s) |
|-------------|-------------------|
| 1           | 3.4               |
| 8           | 21.2              |
| 16          | 42.8              |
| 32          | 85.3              |
