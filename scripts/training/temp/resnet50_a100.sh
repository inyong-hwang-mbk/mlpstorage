#num_acc=("54" "55" "56" "57" "58")
num_acc=("109")

for n in "${num_acc[@]}"; do
  for i in $(seq 0 0); do
    echo "[+] Trial ${i}: ResNet-50 with ${n} acc(s) started."
    RDMAV_FORK_SAFE=1 \
      nocache mlpstorage training run \
	    --hosts localhost \
      --num-client-hosts 1 \
	    --model resnet50 \
      --client-host-memory-in-gb 512 \
	    --accelerator-type a100 \
	    --num-accelerators ${n} \
      --data-dir /mnt/mlperf_storage/training/data \
      --results-dir /home/inyong.hwang/workloads/mlpstorage/training/a100_results \
	    --param dataset.num_files_train=30695 \
	    --param train.epochs=1 \
	    --param train.total_training_steps=200 \
	    --param reader.read_threads=8 \
	    --param reader.computation_threads=8 \
	    --param reader.transfer_size=262144 \
      --param reader.prefetch_size=2 
#      --param reader.data_loader=native_dali
  done
done

echo "[✓] ResNet-50 finished."
