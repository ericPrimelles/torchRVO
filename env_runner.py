import multiprocessing as mp
from multiprocessing.pool import ThreadPool as tp
import subprocess as sp
import shlex
import time

def run_program(cmd):    
    #sp.call(shlex.split(cmd))  # This will block until cmd finishes
    p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = p.communicate()
    return (out, err)


if __name__ == "main":
    process_list = []

    # Create a list of processes
    for i in range(10):
        p = mp.Process(target=run_program, args=("My Program", {"param1": "val1", "param2": i}))
        process_list.append(p)

    # Limit the number of processes running at a time
    max_processes = 2 
    while process_list:
        while len(mp.active_children()) < max_processes and process_list:
            process = process_list.pop()
            process.start()
        
        time.sleep(0.2)

