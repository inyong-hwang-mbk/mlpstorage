#!/bin/bash

#sudo rm -rf /mnt/mlperf_storage/training/a100_data

mlpstorage checkpointing datasize \
  --hosts localhost \
  --model llama3-70b \
  --client-host-memory-in-gb 512 \
  --num-checkpoints-write 1 \
  --num-checkpoints-read 1 \
  --num-processes 64 \
  --checkpoint-folder /mnt/mlperf_storage/checkpointing/llama3_70b \
  --results-dir /home/inyong.hwang/workloads/mlpstorage/checkpointing/results

nocache mlpstorage checkpointing run \
  --hosts localhost \
  --model llama3-70b \
  --client-host-memory-in-gb 512 \
  --num-checkpoints-write 1 \
  --num-checkpoints-read 1 \
  --num-processes 64 \
  --checkpoint-folder /mnt/mlperf_storage/checkpointing/llama3_70b \
  --results-dir /home/inyong.hwang/workloads/mlpstorage/checkpointing/results
