

#%%
from pathlib import Path
from astropy.table import Table, vstack
from astropy.io import ascii
from ezphot.helper import Helper
from astropy.time import Time
import multiprocessing as mp
from tqdm import tqdm
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class calibrationimagecheck(Helper):
    """
    Check if the calibration image is valid.
    """

    def __init__(self):
        super().__init__()
        self._cached_masterframe_tbl = None
        
    def load_masterframe_info(self):

        # Load summary tables
        masterframe_summary_path = Path(self.config['CALIBDATA_MASTERDIR']) / 'summary.ascii_fixed_width'
        if masterframe_summary_path.exists():
            self._cached_masterframe_tbl = ascii.read(masterframe_summary_path, format='fixed_width')
        else:
            print(f"[WARNING] Master frame summary file not found: {masterframe_summary_path}")
            return
        
    def check_flat(self):
        """
        Check if the flat field images are valid.
        """
        self.load_masterframe_info()
        flat_tbl = self._cached_masterframe_tbl[self._cached_masterframe_tbl['imagetyp'] == 'FLAT']
        
        flat_tbl_groups = flat_tbl.group_by(['filtername', 'telname']).groups
        
        for flat_tbl_group in flat_tbl_groups:
            flat_tbl_group.sort('obsdate')
            
        print(f"Found {len(flat_tbl)} valid flat field images.")
        print(flat_tbl)
         
    def process_group(self, args):
        group, max_diff_threshold_init, adaptive_threshold, analyze_filter = args
        from astropy.io import fits
        import numpy as np
        from astropy.time import Time
        from pathlib import Path

        group.sort('obsdate')
        group_flat = group[(group['imagetyp'] == 'FLAT') & (group['filtername'] == analyze_filter)]

        group_ids = []
        current_group_id = 0
        ref_flat = None
        exceed_count = 0
        diff_history = []
        max_diff_threshold = max_diff_threshold_init

        for idx, row in enumerate(group_flat):
            flat_path = Path(row['file'])
            if not flat_path.exists():
                group_ids.append(-1)
                continue

            if 'group_id' in row.colnames and row['group_id'] not in [None, -1]:
                group_ids.append(row['group_id'])
                current_group_id = row['group_id']
                continue

            try:
                with fits.open(flat_path, memmap=True) as hdul:
                    data = hdul[0].data.astype(np.float32)
            except Exception as e:
                print(f"  [SKIP] Failed to read {flat_path.name}: {e}")
                group_ids.append(-1)
                continue

            if ref_flat is None:
                ref_flat = data
                group_ids.append(current_group_id)
                continue

            ratio = data / ref_flat
            rms = np.sqrt(np.mean(ratio**2))
            difference = np.abs(1 - rms)
            print(f"  [INFO] {flat_path.name} - Difference: {difference:.6f}, Threshold: {max_diff_threshold:.6f}")

            if adaptive_threshold and len(diff_history) >= 10 and len(diff_history) % 10 == 0:
                mean_val = np.mean(diff_history)
                std_val = np.std(diff_history)
                updated_threshold = max(max_diff_threshold_init, mean_val)
                max_diff_threshold = updated_threshold
                print(f"  [INFO] Updated max_diff_threshold: {max_diff_threshold:.6f}")

            if difference > max_diff_threshold:
                time_difference = (Time(row['obsdate']) - Time(group_flat['obsdate'][idx - 1])).to_value('jd') if idx > 0 else 0
                if time_difference > 15:
                    current_group_id += 1
                    ref_flat = data
                    exceed_count = 0
                    group_ids.append(current_group_id)
                    print(f"  [INFO] {flat_path.name} exceeds threshold. New group ID: {current_group_id}, Time difference: {time_difference:.2f} days")
                else:
                    exceed_count += 1
                    group_ids.append(-2)
                    print(f"  [WARNING] {flat_path.name} exceeds threshold. Current group ID: {current_group_id}, Exceed count: {exceed_count}")
            else:
                exceed_count = 0
                diff_history.append(difference)
                group_ids.append(current_group_id)

            if exceed_count >= 3:
                current_group_id += 1
                ref_flat = data
                exceed_count = 0
                for back_idx in [-3, -2, -1]:
                    group_ids[back_idx] = current_group_id
                    
        group_flat['group_id'] = group_ids
        
        # 2. Build obsdate range for each valid group_id
        from collections import defaultdict
        import numpy as np

        group_obsdates = defaultdict(list)
        for row, gid in zip(group_flat, group_flat['group_id']):
            if gid >= 0:
                group_obsdates[gid].append(Time(row['obsdate']).jd)

        # Get sorted group_ids and obsdate boundaries
        sorted_group_ids = sorted(group_obsdates.keys())
        group_ranges = []

        for i, gid in enumerate(sorted_group_ids):
            start_jd = np.min(group_obsdates[gid])
            if i == 0:
                lower_bound = -np.inf
            else:
                lower_bound = np.min(group_obsdates[sorted_group_ids[i]])

            if i + 1 < len(sorted_group_ids):
                upper_bound = np.min(group_obsdates[sorted_group_ids[i + 1]])
            else:
                upper_bound = np.inf

            group_ranges.append((gid, lower_bound, upper_bound))
            
        # 3. Propagate group_id to other images in same telescope group
        group['group_id'] = -1  # reset all to -1
        for row in group_flat:
            group['group_id'][group['file'] == row['file']] = row['group_id']  # apply group_id for analyzed filter only

        for gid, start_jd, end_jd in group_ranges:
            for idx, row in enumerate(group):
                jd = Time(row['obsdate']).jd
                if start_jd <= jd < end_jd and group['group_id'][idx] == -1:
                    group['group_id'][idx] = gid  # propagate to bias/dark/other filters

        a = group.group_by('group_id').groups
        
        return group

    def analyze_flat_patterns_memory_safe(
        self,
        max_diff_threshold_init=0.0005,
        min_group_size=3,
        adaptive_threshold=True,
        analyze_filter: str = 'g',
        n_process=4,
        update: bool = True
        ):
        self.load_masterframe_info()
        if 'group_id' not in self._cached_masterframe_tbl.colnames:
            self._cached_masterframe_tbl['group_id'] = -1
            
        tel_groups = self._cached_masterframe_tbl.group_by(['telkey', 'telname']).groups

        # Filter out small groups
        task_list = [
            (group.copy(), max_diff_threshold_init, adaptive_threshold, analyze_filter)
            for group in tel_groups
        ]

        print(f"[MULTIPROCESS] Launching {len(task_list)} groups with {n_process} processes")

        with mp.get_context("fork").Pool(processes=n_process) as pool:
            results = []
            with tqdm(total=len(task_list), desc="Processing groups") as pbar:
                for result in pool.imap_unordered(self.process_group, task_list):
                    results.append(result)
                    pbar.update(1)

        final_groups = Table()
        # Merge results back into _cached_masterframe_tbl
        for group in results:
            final_groups = vstack([final_groups, group])
        print("[DONE] Flat grouping complete.")
        self._cached_masterframe_tbl = final_groups
        
        if update:
            self._cached_masterframe_tbl.write(
                Path(self.config['CALIBDATA_MASTERDIR']) / 'summary.ascii_fixed_width',
                format='ascii.fixed_width',
                overwrite=True
            )
        print(f"[UPDATE] Updated master frame summary table with {len(final_groups)} entries.")
        

#%%
if __name__ == "__main__":
    self = calibrationimagecheck()
    max_diff_threshold_init=0.0005
    min_group_size=3
    adaptive_threshold=True
    n_process=20
    self.analyze_flat_patterns_memory_safe(
        max_diff_threshold_init=max_diff_threshold_init,
        min_group_size=min_group_size,
        adaptive_threshold=adaptive_threshold,
        analyze_filter='g',
        n_process=n_process,
        update=True
    )
    # self._cached_masterframe_tbl['group_id'] = -1
    # self._cached_masterframe_tbl.write(
    #     Path(self.config['CALIBDATA_MASTERDIR']) / 'summary.ascii_fixed_width',
    #     format='ascii.fixed_width',
    #     overwrite=True
    # )
    
#%%
