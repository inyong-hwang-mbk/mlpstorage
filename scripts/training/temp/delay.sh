delay=("0" "10")

#bash setup.sh

for d in "${delay[@]}"; do
  #for i in $(seq 0 7); do
    #ssh -t inyong.hwang@10.1.5.52 "sudo /home/inyong.hwang/spdk/scripts/rpc.py bdev_delay_update_latency DelayNvme${i}n1 avg_read ${d}"
    #ssh -t inyong.hwang@10.1.5.52 "sudo /home/inyong.hwang/spdk/scripts/rpc.py bdev_delay_update_latency DelayNvme${i}n1 p99_read ${d}"
  #done
  bash unet_3d.sh ${d}
  #bash cosmoflow.sh ${d}
  #bash resnet50.sh ${d}
done
