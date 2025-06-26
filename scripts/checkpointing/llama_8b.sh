#!/bin/bash

#sudo rm -rf /mnt/mlperf_storage/training/a100_data

mlpstorage checkpointing datasize \
  --hosts localhost \
  --model llama3-8b \
  --client-host-memory-in-gb 512 \
  --num-checkpoints-write 10 \
  --num-checkpoints-read 10 \
  --num-processes 8 \
  --checkpoint-folder /mnt/mlperf_storage/checkpointing/llama3_8b \
  --results-dir /home/inyong.hwang/workloads/mlpstorage/checkpointing/results

mlpstorage checkpointing run \
  --hosts localhost \
  --model llama3-8b \
  --client-host-memory-in-gb 512 \
  --num-checkpoints-write 10 \
  --num-checkpoints-read 10 \
  --num-processes 8 \
  --checkpoint-folder /mnt/mlperf_storage/checkpointing/llama3_8b \
  --results-dir /home/inyong.hwang/workloads/mlpstorage/checkpointing/results \
  --param reader.odirect=1
  
