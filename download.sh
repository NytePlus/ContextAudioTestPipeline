
dcsync -u wangchencheng -d s2 -s "/aistor/sjtu/hpc_stor01/home/wangchencheng/models/Qwen/Qwen2___5-Omni-7B" -t /hpc_stor03/sjtu_home/chencheng.wang/models/ --download 
# 这个集群clone仓库也太慢了

docker build -f Dockerfile.qwen_asr -t hub.szaic.com/sjtu/sjtu_yukai-wangchencheng_qwenasr:v1.1 .

docker run -it --rm -e ASCEND_VISIBLE_DEVICES=0-7 \
 -v /hpc_stor03/sjtu_home/chencheng.wang/models:/models \
 -v /hpc_stor03/sjtu_home/chencheng.wang/workspace/ContextAudioTestPipeline:/workspace \
 docker.v2.aispeech.com/sjtu/sjtu_yukai-yiweiguo-metric:v1.0

modelscope download --model Qwen/Qwen2.5-Omni-7B

vc submit -p pdgpu-sjtu-ai -i hub.szaic.com/sjtu/sjtu_yukai-wangchencheng_qwenasr:v1.1 -c 160 -m 960G -g 8 -j "qwen_asr" -v "/aistor/sjtu/hpc_stor01/home/wangchencheng/workspace/AudioOmniTest:/workspace,/aistor/sjtu/hpc_stor01/home/wangchencheng/models:/models,/aistor/sjtu/hpc_stor01/home/wangchencheng/data:/data" \
    -d $PWD JOB=1:1 log/slurm.log --cmd "sleep 10000000"

export PYTHONPATH="/workspace:/workspace/pipelines/LongCatFlashOmni:/workspace/pipelines/Ming:/workspace/pipelines/DeSTA25_Audio:$PYTHONPATH"