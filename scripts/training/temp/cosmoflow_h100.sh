#num_acc=("20" "21" "22" "23" "24")
num_acc=("27")

for n in "${num_acc[@]}"; do
  for i in $(seq 0 0); do
    echo "[+] Trial ${i}: Cosmoflow with ${n} acc(s) started."
    RDMAV_FORK_SAFE=1 \
    nocache mlpstorage training run \
	    --hosts localhost \
      --num-client-hosts 1 \
	    --model cosmoflow \
      --client-host-memory-in-gb 512 \
	    --accelerator-type h100 \
	    --num-accelerators ${n} \
      --data-dir /mnt/mlperf_storage/training/data \
	    --results-dir /home/inyong.hwang/workloads/mlpstorage/training/h100_results \
	    --param dataset.num_files_train=971819 \
	    --param train.epochs=1 \
	    --param train.total_training_steps=5000 \
	    --param reader.read_threads=2 \
	    --param reader.computation_threads=2 \
	    --param reader.transfer_size=262144 \
	    --param reader.prefetch_size=2
  done
done

echo "[✓] Cosmoflow finished."
