import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import subprocess as sp
import shlex
import pandas as pd
import sys

class Run:
    def __init__(self, pid: int, epochs: int, episodes: int,
                  frame_count: int, buffer_len: int, batch_size: int, train: int) -> None:
        self.pid = pid
        self.epochs = epochs
        self.episodes = episodes
        self.frame_count = frame_count
        self.buffer_len = buffer_len
        self.batch_size = batch_size
        self.train = train

    def get_args(self):
        return f'{self.frame_count} {self.buffer_len} {self.batch_size} {self.epochs} {self.episodes} {self.train} {self.pid}'


def parse_csv(filename: str)-> list[Run]:
    runs: list[Run] = []
    df: pd.DataFrame = pd.read_csv(filename)
    for _, row in df.iterrows():
        runs.append(Run(row['pid'], row['epochs'], row['episodes'], 
            row['frame_count'], row['buffer_len'], row['batch_size'], row['train']))
    return runs 

def call_proc(cmd):    
    p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = p.communicate()
    return (out, err)

if __name__ == "__main__":
    # parámetros: filename
    filename = sys.argv[1]
    # runs
    runs = parse_csv(filename)
    results = []
    pool = ThreadPool(mp.cpu_count())
    process_list = []
    # Create a list of processes
    for run in runs:
        cmd = f'python3 MADDPG.py {run.get_args()}'
        results.append(pool.apply_async(call_proc, args=(cmd,)))
    
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        print("out: {} err: {}".format(out, err))
    
