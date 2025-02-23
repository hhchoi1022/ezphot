

#%%
import numpy as np
from multiprocessing import Pool
import os
from astropy.table import Table

from tippy.helper import Helper
from tippy.configuration import TIPConfig
from tippy.image import CalibrationImage, ScienceImage 
from tippy.routine import SDTObsdata
from tippy.routine import Preprocess

class SDT_Routine(TIPConfig):
    
    def __init__(self, max_processes = 5):
        self.telinfo = None
        self.max_processes = max_processes
        self.helper = Helper()
        

    def _divide_imgtype_worker(self, args):
        """Worker function for parallel processing (each telescope gets one process)."""
        telescope_id, filelist = args
        imginfo = self.helper.get_imginfo(filelist=filelist, normalize_key=True)
        imginfo['imgtype'] = np.char.upper(imginfo['imgtype'])

        # Define classification keys
        imgtype_dict = {
            'object': ['LIGHT', 'OBJECT', 'TARGET', 'SCIENCE'],
            'bias': ['BIAS', 'ZERO'],
            'dark': ['DARK'],
            'flat': ['FLAT']
        }

        # Create dictionaries to store classified data
        target_data_by_type = {key: [] for key in imgtype_dict.keys()}
        imginfo_by_type = {key: None for key in imgtype_dict.keys()}

        # Apply classification
        for category, keys in imgtype_dict.items():
            idx = np.isin(imginfo['imgtype'], keys)
            target_data_by_type[category] = np.array(filelist)[idx]
            
            # Store classified imginfo as an Astropy Table
            filtered_info = {k: v[idx] for k, v in imginfo.items()}
            imginfo_by_type[category] = Table(filtered_info) if len(filtered_info['imgtype']) > 0 else Table()

        return telescope_id, target_data_by_type, imginfo_by_type

    def divide_imgtype(self, filelist_dict):
        """Run image classification in parallel (one process per telescope_id)."""
        telescope_ids = list(filelist_dict.keys())
        filelists = list(filelist_dict.values())

        with Pool(processes=min(self.max_processes, len(telescope_ids))) as pool:
            results = pool.map(self._divide_imgtype_worker, zip(telescope_ids, filelists))

        # Merge results into dictionaries
        target_data_by_type = {telescope_id: data for telescope_id, data, _ in results}
        imginfo_by_telescope = {telescope_id: imginfo for telescope_id, _, imginfo in results}

        return target_data_by_type, imginfo_by_telescope


    def _preprocess_worker(self, args):
        from tippy.routine import Preprocess
        """ Worker function to generate master frames if calibration frames are present. """
        telescope_id, filelist_by_type = args
        preprocess = Preprocess()
        
        calib_path = os.path.join(self.config['CALIBDATA_PATH'], self.telname, telescope_id)

        # Output directories
        bias_dir = os.path.join(calib_path, "bias")
        dark_dir = os.path.join(calib_path, "dark")
        flat_dir = os.path.join(calib_path, "flat")
        os.makedirs(bias_dir, exist_ok=True)
        os.makedirs(dark_dir, exist_ok=True)
        os.makedirs(flat_dir, exist_ok=True)
        
        # Extract calibration frames
        bias_files = filelist_by_type.get("bias", [])
        dark_files = filelist_by_type.get("dark", [])
        flat_files = filelist_by_type.get("flat", [])
        
        # Convert to CalibrationImage objects
        bias_images = [CalibrationImage(path=f, telinfo=self.telinfo) for f in bias_files] if len(bias_files)>0 else []
        dark_images = [CalibrationImage(path=f, telinfo=self.telinfo) for f in dark_files] if len(dark_files)>0 else []
        flat_images = [CalibrationImage(path=f, telinfo=self.telinfo) for f in flat_files] if len(flat_files)>0 else []

        # Search for nearest master bias if no bias frames are available
        mbias = None
        if len(bias_images) == 0:
            #################################### IF NO BIAS FRAMES, SEARCH BIAS WITH PREPROCESS CLASS AND REGISTER IT
            print(f"No bias frames for {telescope_id}, searching for nearest master bias...")
            bias_candidates = sorted(glob.glob(os.path.join(bias_dir, "*.fits")))
            if bias_candidates:
                mbias = [CalibrationImage(path=bias_candidates[-1], telinfo=self.telname)]
                print(f"Using nearest master bias: {mbias[0].path}")
                
        # Use available dark frames or None
        mdark = dark_images if dark_images else None

        # If there are calibration images, generate master frames
        if bias_images or dark_images or flat_images:
            print(f"Generating master frames for {telescope_id}...")
            master_frames = preprocess.generate_master_frame(
                calib_imagelist=bias_images + dark_images + flat_images,
                mbias=mbias,
                mdark=mdark
            )
            return telescope_id, master_frames
        else:
            print(f"No calibration frames found for {telescope_id}. Skipping master frame generation.")
            return telescope_id, {}
        
    def register_master_frame(self, imginfo_by_type):
        """ Run preprocessing in parallel (one process per telescope_id). """
        telescope_ids = list(target_data_by_type.keys())

        with Pool(processes=min(self.max_processes, len(telescope_ids))) as pool:
            results = pool.map(self._preprocess_worker, [(tid, imginfo_by_type[tid]) for tid in telescope_ids])

        # Merge results into a dictionary
        return {telescope_id: data for telescope_id, data in results}
        
    def main(self, foldername: str):
        # First, check if the number of files in the source and destination directories are the same
        sdtdata = SDTObsdata()
        obsdata = sdtdata.show_obsdata(foldername, show_only_numbers=True)
        destdata = sdtdata.show_destdata(foldername, show_only_numbers=True)

        # If the number of files are different, sync the data
        if not np.sum(list(obsdata.values())) == np.sum(list(destdata.values())):
            print('The number of files in the source and destination directories are different.')
            print('Start syncing the data...')
            sdtdata.sync_obsdata(foldername)

        target_data = sdtdata.show_destdata(foldername, show_only_numbers=False)

        # Run divide_imgtype in parallel 
        target_data_by_type, imginfo_by_type = self.divide_imgtype(target_data)
        
        # Generate master frames if calibration frames are present. Else, register the masterframe for each telescope. 
        target_data_by_type, imginfo_by_type = self.register_master_frame(imginfo_by_type)
        
        # Assign master frames to all images.
        return target_data_by_type

#%%
from tippy.routine import SDTObsdata
self = SDT_Routine()
foldername = '2025-02-02_gain2750'

# %%
