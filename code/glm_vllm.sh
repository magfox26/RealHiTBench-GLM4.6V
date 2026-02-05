CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /export/home/xwang/data1/liu/models/GLM-4.6V \
    --tensor-parallel-size 4 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --gpu-memory-utilization 0.96 \
    --max-num-seqs 16 \