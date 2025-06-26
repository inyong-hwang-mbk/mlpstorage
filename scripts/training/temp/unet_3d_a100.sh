#!/bin/bash
#num_acc=("1" "2" "4" "8" "16")
num_acc=("33" "35" "36")

for n in "${num_acc[@]}"; do
  for i in $(seq 0 0); do
    echo "[+] Trial ${i}: 3D U-Net with ${n} acc(s) started."
    RDMAV_FORK_SAFE=1 \
      mlpstorage training run \
	    --hosts localhost \
      --num-client-hosts 1 \
      --client-host-memory-in-gb 512 \
      --num-accelerators ${n} \
      --accelerator-type a100 \
      --model unet3d \
      --data-dir /mnt/mlperf_storage/training/data \
      --checkpoint-folder /mnt/mlperf_storage/training/checkpoint \
      --results-dir /home/inyong.hwang/workloads/mlpstorage/training/a100_results \
	    --param dataset.num_files_train=45500 \
	    --param train.epochs=1 \
	    --param train.total_training_steps=100 \
	    --param reader.read_threads=8 \
	    --param reader.computation_threads=8 \
	    --param reader.transfer_size=262144 \
	    --param reader.prefetch_size=2 \
      --param reader.odirect=1
  done
done

echo "[✓] 3D U-Net finished."

