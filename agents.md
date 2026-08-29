# 在这台服务器上跑 GLM-5.3-Flash (华为的910B 8卡 /npu)
# 目前是 vllm已经有glm-5.3-专用的镜像了,但他是 nvidia卡,还没有支持910b的,
# 连接到服务器要通过跳板机
ssh -i ~/.ssh/id_rsa_2 -o StrictHostKeyChecking=no 192.168.130.55 \
  "ssh -o StrictHostKeyChecking=no   root@172.24.129.105 '<CMD>'"
## 查日志:
## 目前已经做了一些改进还有错:

ssh -i ~/.ssh/id_rsa_2 -o BatchMode=yes 192.168.130.55 "ssh -o BatchMode=yes root@172.24.129.106 'docker exec glm53-serve bash -c \"grep -n KVDBG /data/glm53-serve.log | head; echo ===; tail -n 3 /data/glm53-serve.log\"'"
