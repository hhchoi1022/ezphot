
#%%
import subprocess
from pathlib import Path

class Tract7DTWrapper:

    def __init__(self,
                 conda_env="tract7dt",
                 python_exe="python"):
        
        self.conda_env = conda_env
        self.python_exe = python_exe

    def run(self,
            config_path: Path,
            log_path: Path = None,
            verbose: bool = True):

        cmd = [
            "conda", "run", "-n", self.conda_env,
            self.python_exe,
            "-m", "tract7dt",   # 또는 entrypoint script
            str(config_path)
        ]

        if verbose:
            print("[RUN]", " ".join(cmd))

        if log_path is not None:
            with open(log_path, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=f, check=True)
        else:
            subprocess.run(cmd, check=True)
            
#%%