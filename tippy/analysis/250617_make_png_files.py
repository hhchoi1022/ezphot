
#%%
from tippy.utils import SDTData
from tippy.helper import Helper
from tippy.image import ScienceImage
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import threading
import time
from multiprocessing import Pool
#%%

target_namelist = ['T00138','T00139','T00174','T00175','T00176','T00215','T00216']
sdtdata = SDTData()
helper = Helper()

#%%
# Memory monitor
def periodic_memory_report(interval=10):
    def loop():
        while True:
            helper.report_memory_process()
            time.sleep(interval)
    t = threading.Thread(target=loop, daemon=True)
    t.start()

periodic_memory_report(10)

# Image processor
def process_single_image(target_path, telinfo):
    S = ScienceImage(target_path, telinfo, load=True)
    if Path(str(S.savepath.savepath) + '.png').exists():
        del S
        gc.collect()
        return
    fig, ax = S.show(title=S.path.name, save_path=str(S.savepath.savepath) + '.png', close_fig=True)
    plt.close(fig)
    S.data = None
    del S, fig, ax
    gc.collect()

def process_single_image_wrapper(args):
    return process_single_image(*args)

# Build full task list
task_list = []

for target_name in tqdm(target_namelist, desc="Gathering tasks"):
    try:
        targetpath_dict = sdtdata.show_scisourcedata(
            targetname=target_name,
            show_only_numbers=False,
            exclude_combined=True,
            key='filter'
        )
        # Use the first available filter's path to estimate telescope info
        first_paths = next(iter(targetpath_dict.values()))
        if not first_paths:
            continue
        telinfo = helper.estimate_telinfo(first_paths[0])

        for target_paths in targetpath_dict.values():
            task_list.extend([(path, telinfo) for path in target_paths])
    except Exception as e:
        print(f"Error with target {target_name}: {e}")
        continue
#%%
# Process all images in parallel
if __name__ == "__main__":
    with Pool(processes=15) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_image_wrapper, task_list), total=len(task_list), desc="Processing"):
            pass