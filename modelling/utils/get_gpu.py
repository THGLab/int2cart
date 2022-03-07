

from io import StringIO
import subprocess
import pandas as pd


def handle_gpu(gpu_requirements):
    '''
    Return the first n free GPU that has enough resource for running a job

    args:
        count = the total number of returned free GPUs
    '''
    if type(gpu_requirements) is not list:
        if "cuda:" in gpu_requirements or gpu_requirements != "cuda":
            return [gpu_requirements]
        else:
            try:
                gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
            except:
                return []
            gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8").replace("MiB","")),
                                names=['memory.used', 'memory.free'],
                                skiprows=1)
            gpu_df["usage"]=gpu_df["memory.free"]/(gpu_df["memory.used"]+gpu_df["memory.free"])
            idxes = list(gpu_df.sort_values("usage", ascending=False).index)
            for index in idxes.copy():
                if gpu_df.loc[index,"usage"] < 0.1:
                    idxes.remove(index)
            if len(idxes) >= 1:
                return ["cuda:" + str(idxes[0])]
            else:
                raise RuntimeError("No available GPUs!")
    else:
        return gpu_requirements
