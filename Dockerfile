FROM vllm-ascend:glm53-port-tritonfix

# GLM-5.3-Flash KDA sigmoid fix
# Base: vllm-ascend:glm53-port-tritonfix (vLLM-Ascend 0.19.1rc2.dev1713+g128ecf890)
# Fix: add torch.sigmoid(beta) in AscendKimiGatedDeltaNetAttention._forward
#       Without sigmoid, beta ranged [-1, 2.4] instead of [0, 1] -> garbage output

COPY glm5_next_patches/kda_ascend.py /vllm-workspace/vllm-ascend/vllm_ascend/models/glm5_next/kda_ascend.py
COPY glm5_next_patches/glm5_next_model.py /vllm-workspace/vllm-ascend/vllm_ascend/models/glm5_next/model.py

# Default serve command (override at runtime)
CMD ["sleep", "infinity"]
