#!/bin/bash
workloads=("unet3d" "cosmoflow" "resnet50")
#workloads=("resnet50")

declare -A num_acc_map
num_acc_map=(["unet3d"]="20" ["cosmoflow"]="192" ["resnet50"]="192")

declare -A num_files_map
num_files_map=(["unet3d"]="70000" ["cosmoflow"]="971819" ["resnet50"]="30695")


for w in "${workloads[@]}"; do
  echo "[+] Generate dataset for ${w}..."
  mlpstorage training datasize \
    --model ${w} \
    --accelerator-type h100 \
    --max-accelerators ${num_acc_map[$w]} \
    --data-dir /mnt/mlperf_storage/training/data \
    --results-dir /home/inyong.hwang/Evaluation/workloads/mlpstorage/training/h100_results \
    --num-client-hosts 1 \
    --client-host-memory-in-gb 512
  
  nocache mlpstorage training datagen \
    --hosts=localhost \
    --model=${w} \
    --exec-type=mpi \
    --num-processes=128 \
    --data-dir=/mnt/mlperf_storage/training/data \
    --results-dir=/home/inyong.hwang/Evaluation/workloads/mlpstorage/training/h100_results \
    --param dataset.num_files_train=${num_files_map[$w]} 

done

echo "[✓] Dataset Generation Completed."

