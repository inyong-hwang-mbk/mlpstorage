import numpy as np
import os
import json
import signal
import itertools
import parse
import subprocess
import threading
import sys
import shutil

from subprocess import Popen, PIPE, STDOUT

RESULTS_DIR = "/home/inyong.hwang/Evaluation/workloads/mlpstorage"

class Evaluator:
    def __init__(self, transport, trial, criteria):
        self.baselines = {}
        with open(f"baselines/{transport}_A100.json", "r") as f:
            self.baselines['a100'] = json.load(f)
        with open(f"baselines/{transport}_H100.json", "r") as f:
            self.baselines['h100'] = json.load(f)
        self.trial = trial
        self.criteria = criteria

    def evaluate(self, model, acc_type):
        cur_acc = self.baselines[acc_type][model]['num-accelerators']
        min_acc = 1
        max_acc = 2 * cur_acc
        flag = 0
        score = 0
        print(f'[+] Starting {model} score evaluation on {acc_type}.')
        print(f'    - range = [{min_acc} ~ {max_acc}] (started from {cur_acc} acc(s))')
        
        shutil.rmtree(f"{RESULTS_DIR}/{acc_type}_results")
        while True:
            print(f'[*] {cur_acc} accelerators test.')
            au_list = self._evaluate(model, acc_type, cur_acc)

            au_list = np.array(au_list)
            if self.criteria == 'avg':
                metric = np.mean(au_list)
            elif self.criteria == 'max':
                metric = np.max(au_list)

            if model == 'cosmoflow':
                if metric >= 70:
                    flag = 1
                else: flag = 0
            else:
                if metric >= 90:
                    flag = 1
                else: flag = 0

            if flag == 1: # Pass
                print(f'    Passed :) (AU: {metric:.2f}%)')
                min_acc = cur_acc 
                cur_acc = (max_acc + cur_acc) // 2
            else: # Fail
                print(f'    Failed :( (AU: {metric:.2f}%)')
                max_acc = cur_acc - 1
                cur_acc = (min_acc + cur_acc - 1) // 2
                
            if min_acc == max_acc - 1 or min_acc == max_acc:
                score = min_acc
                break
        
        self.baselines[acc_type][model]['num-accelerators'] = score
        print(f'[✓] {model} score on {acc_type} is {score}.')
    
    def run(self, model, acc_type, num_acc):
        p = self.baselines[acc_type][model]
        pp = p['param']
        results_dir = f"{RESULTS_DIR}/{acc_type}_results/{num_acc}_acc" 

        self._run(p['num-client-hosts'], model, p['client-host-memory-in-gb'], acc_type,
            num_acc, p['data-dir'], results_dir, pp['dataset.num_files_train'], pp['train.epochs'],
            pp['train.total_training_steps'], pp['reader.read_threads'], pp['reader.computation_threads'],
            pp['reader.transfer_size'], pp['reader.prefetch_size'])

    def record(self, transport):
        with open(f"baselines/{transport}_A100.json", "w") as f:
            json.dump(self.baselines['a100'], f)
        with open(f"baselines/{transport}_H100.json", "w") as f:
            json.dump(self.baselines['h100'], f)

        print('-----------------------------------------------')
        print(f'[✓] Updated baselines have been saved.')
        print(f'    Path: {os.getcwd()}/baselines')

    def _evaluate(self, model, acc_type, cur_acc):
        # self.trial 만큼 mlpstorage training run
        for i in range(int(self.trial)):
            self.run(model, acc_type, cur_acc)
        
        # Result AU 수집
        au_list = []
        results_dir = f"{RESULTS_DIR}/{acc_type}_results/{cur_acc}_acc/training/{model}/run"
        for subdir in os.listdir(results_dir):
            with open(f"{results_dir}/{subdir}/summary.json", "r") as f:
                summary = json.load(f)
                au_list.append(summary['metric']['train_au_percentage'][0])
        shutil.rmtree(f"{RESULTS_DIR}/{acc_type}_results/{cur_acc}_acc")

        return au_list


    def _run(self, num_client_hosts, model, client_host_memory_in_gb, accelerator_type, num_accelerators,
            data_dir, results_dir, num_files_train, epochs, total_training_steps, read_threads,
            computation_threads, transfer_size, prefetch_size):
        cmd = [
            'nocache',
            'mlpstorage', 'training', 'run',
            '--hosts', 'localhost',
            '--num-client-hosts', str(num_client_hosts),
            '--model', model,
            '--client-host-memory-in-gb', str(client_host_memory_in_gb),
            '--accelerator-type', accelerator_type,
            '--num-accelerators', str(num_accelerators),
            '--data-dir', data_dir,
            '--results-dir', results_dir,
            '--param', f'dataset.num_files_train={num_files_train}',
            '--param', f'train.epochs={epochs}',
            '--param', f'train.total_training_steps={total_training_steps}',
            '--param', f'reader.read_threads={read_threads}',
            '--param', f'reader.computation_threads={computation_threads}',
            '--param', f'reader.transfer_size={transfer_size}',
            '--param', f'reader.prefetch_size={prefetch_size}',
        ]
        # if model == 'unet3d':
        #   cmd.pop(0)
        #   cmd.append('--checkpoint-folder')
        #   cmd.append('/mnt/mlperf_storage/training/checkpoint')
        #   cmd.append('--param')
        #   cmd.append('reader.odirect=1')
        env = os.environ.copy()
        env['RDMA_FORK_SAFE'] = '1'
        #env['OMPI_MCA_hwloc_base_binding_policy'] = 'hwthread'
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    eval = Evaluator(sys.argv[1], sys.argv[2], sys.argv[3])
    #eval.evaluate('cosmoflow', 'h100')
    eval.evaluate('resnet50', 'h100')
    eval.record(sys.argv[1])

if __name__ == '__main__':
    main()
