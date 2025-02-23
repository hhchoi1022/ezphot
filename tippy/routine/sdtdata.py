#%%
import subprocess
import glob
import os
import shutil
from multiprocessing import Pool
from tippy.configuration import TIPConfig
from astropy.io import fits
import subprocess
import glob
import os
from multiprocessing import Pool
from astropy.table import Table
from tippy.configuration import TIPConfig
from tippy.helper import Helper

#%%

class SDTData(TIPConfig):
    def __init__(self, 
                 ccd : str = 'C361K'):
        """
        Initializes the ObsDataSync class.
        
        :param source_base: Base source directory containing telescope folders (7DT??)
        :param dest_base: Base destination directory where data will be synced
        """
        super().__init__()
        self.folders = []
        self.ccd = ccd
        self.helper = Helper()

    def show_obssourcedata(self, foldername: str, show_only_numbers : bool = False):
        """
        Shows the number of FITS files for each specified folder across all 7DT?? directories.
        
        :param foldername: List of folder names or a single folder name (string).
        :return: Dictionary sorted by folder name, and within each folder, sorted by telescope ID.
        """
        # Ensure foldername is a list

        fits_counts = {}

        # Find all telescope directories matching 7DT??
        telescope_dirs = glob.glob(os.path.join( self.config['SDTDATA_OBSSOURCEDIR'], "7DT??"))

        for telescope_dir in telescope_dirs:
            telescope_id = os.path.basename(telescope_dir)  # Extract 7DTxx name

            folder_path = os.path.join(telescope_dir, foldername)

            # Count FITS files if the folder exists
            if os.path.isdir(folder_path):
                fits_files = glob.glob(os.path.join(folder_path, "*.fits"))
                if show_only_numbers:
                    fits_counts[telescope_id] = len(fits_files)
                else:
                    fits_counts[telescope_id] = sorted(fits_files)

        # Sort by telescope ID
        sorted_fits_counts = {tid: fits_counts[tid] for tid in sorted(fits_counts)}
        
        if not sorted_fits_counts:
            print("No matching folders found.")
            return sorted_fits_counts
        else:
            print(sorted_fits_counts)
            return sorted_fits_counts
        
    def show_obsdestdata(self, foldername : str, show_only_numbers : bool = False):
        """
        Shows the number of FITS files for each specified folder across all 7DT?? directories.
        
        :param foldername: List of folder names or a single folder name (string).
        :return: Dictionary sorted by folder name, and within each folder, sorted by telescope ID.
        """
        # Ensure foldername is a list

        fits_counts = {}

        # Find all telescope directories matching 7DT??
        telescope_dirs = glob.glob(os.path.join( self.config['SDTDATA_OBSDESTDIR'], '7DT*', foldername))

        for telescope_dir in telescope_dirs:
            telescope_id = os.path.basename(os.path.dirname(telescope_dir))
            fits_files = glob.glob(os.path.join(telescope_dir, "*.fits"))
            if show_only_numbers:
                fits_counts[telescope_id] = len(fits_files)
            else:
                fits_counts[telescope_id] = sorted(fits_files)
                    
        # Sort by telescope ID
        sorted_fits_counts = {tid: fits_counts[tid] for tid in sorted(fits_counts)}

        if not sorted_fits_counts:
            print("No matching folders found.")
            return sorted_fits_counts
        else:
            print(sorted_fits_counts)
            return sorted_fits_counts

    def show_obssourcefolder(self, folder_key : str = None):
        """
        Shows the contents of the source and destination directories.
        """
        print("Source directory:", os.path.join( self.config['SDTDATA_OBSSOURCEDIR'], "7DT??", folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        folders = set()

        for entry in os.scandir( self.config['SDTDATA_OBSSOURCEDIR']):
            if entry.is_dir() and entry.name.startswith("7DT") and len(entry.name) == 5:
                subfolders = {os.path.join(sub.name) for sub in os.scandir(entry.path) if sub.is_dir()}
                folders.update(subfolders)
                
        if not folder_key:
            return sorted_folders
        else:
            matched_folders = []
            for folder in folders:
                if folder_key in folder:
                    matched_folders.append(folder)
                else:
                    pass
            if not matched_folders:
                print("No matching folders found.")
            else:
                print("Matching folders:", sorted(matched_folders))
                return sorted(matched_folders)

    def show_obsdestfolder(self, folder_key : str = None):
        """
        Shows the contents of the source and destination directories.
        """
        print("Source directory:", os.path.join( self.config['SDTDATA_OBSDESTDIR'], "7DT??", folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        folders = set()

        for entry in os.scandir( self.config['SDTDATA_OBSDESTDIR']):
            if entry.is_dir() and entry.name.startswith("7DT") and len(entry.name) == 5:
                subfolders = {os.path.join(sub.name) for sub in os.scandir(entry.path) if sub.is_dir()}
                folders.update(subfolders)
                
        if not folder_key:
            return sorted_folders
        else:
            matched_folders = []
            for folder in folders:
                if folder_key in folder:
                    matched_folders.append(folder)
                else:
                    pass
            if not matched_folders:
                print("No matching folders found.")
            else:
                print("Matching folders:", sorted(matched_folders))
                return sorted(matched_folders)
        
    def show_scisourcedata(self, targetname: str, 
                           show_only_numbers : bool = False, 
                           exclude_combined : bool = True,
                           key : str = 'filter' # filter or telescope
                           ):
        """
        Shows the number of FITS files for each specified folder across all 7DT?? directories.
        
        :param foldername: List of folder names or a single folder name (string).
        :return: Dictionary sorted by folder name, and within each folder, sorted by telescope ID.
        """
        # Ensure foldername is a list

        fits_counts = {}

        # Find all telescope directories matching 7DT??
        os.listdir(self.config['SDTDATA_SCISOURCEDIR'])
        
        if key.lower() == 'filter':
            dirs = glob.glob(os.path.join( self.config['SDTDATA_SCISOURCEDIR'], targetname, "7DT??", '*'))
        elif key.lower() == 'telescope':
            dirs = glob.glob(os.path.join( self.config['SDTDATA_SCISOURCEDIR'], targetname, "7DT??"))
        else:
            raise ValueError("Invalid key. Must be 'filter' or 'telescope'.")
        
        for dir in dirs:
            id_ = os.path.basename(dir)  # Extract 7DTxx name

            # Count FITS files if the folder exists
            if key.lower() == 'filter':
                fits_files = glob.glob(os.path.join(dir, "*.fits"))
            else:
                fits_files = glob.glob(os.path.join(dir, "*", "*.fits"))
            
            if exclude_combined:
                fits_files = [f for f in fits_files if ".com." not in f]
                
            if show_only_numbers:
                fits_counts[id_] = len(fits_files)
            else:
                fits_counts[id_] = sorted(fits_files)

        # Sort by telescope ID
        sorted_fits_counts = {id_: fits_counts[id_] for id_ in sorted(fits_counts)}
        
        if not sorted_fits_counts:
            print("No matching folders found.")
            return sorted_fits_counts
        else:
            print(sorted_fits_counts)
            return sorted_fits_counts

    def show_scidestdata(self, targetname : str, 
                         show_only_numbers : bool = False,
                         exclude_combined : bool = False,
                         key : str = 'filter' # filter or telescope
                         ):
        """
        Shows the number of FITS files for each specified folder across all 7DT?? directories.
        
        :param foldername: List of folder names or a single folder name (string).
        :return: Dictionary sorted by folder name, and within each folder, sorted by telescope ID.
        """
        # Ensure foldername is a list

        fits_counts = {}

        # Find all telescope directories matching 7DT??
        if key.lower() == 'filter':
            dirs = glob.glob(os.path.join( self.config['SDTDATA_SCIDESTDIR'], targetname, "7DT??", '*'))
        elif key.lower() == 'telescope':
            dirs = glob.glob(os.path.join( self.config['SDTDATA_SCIDESTDIR'], targetname, "7DT??"))
        else:
            raise ValueError("Invalid key. Must be 'filter' or 'telescope'.")
        
        for dir in dirs:
            id_ = os.path.basename(dir)  # Extract 7DTxx name

            # Count FITS files if the folder exists
            if key.lower() == 'filter':
                fits_files = glob.glob(os.path.join(dir, "*.fits"))
            else:
                fits_files = glob.glob(os.path.join(dir, "*", "*.fits"))
            
            if exclude_combined:
                fits_files = [f for f in fits_files if ".com." not in f]
                
            if show_only_numbers:
                fits_counts[id_] = len(fits_files)
            else:
                fits_counts[id_] = sorted(fits_files)
                    
        # Sort by telescope ID
        sorted_fits_counts = {id_: fits_counts[id_] for id_ in sorted(fits_counts)}

        if not sorted_fits_counts:
            print("No matching folders found.")
            return sorted_fits_counts
        else:
            print(sorted_fits_counts)
            return sorted_fits_counts

    def show_scisourcefolder(self, folder_key : str = '*'):
        """
        Shows the contents of the source and destination directories.
        """
        print("Source directory:", os.path.join( self.config['SDTDATA_SCISOURCEDIR'], folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        matched_folders = []
        all_targets = os.listdir(self.config['SDTDATA_SCISOURCEDIR'])
        for target in all_targets:
            if folder_key in target:
                matched_folders.append(target)
        all_matched_folders = set(matched_folders)
        return sorted(all_matched_folders)

    def show_obsdestfolder(self, folder_key : str = '*'):
        """
        Shows the contents of the source and destination directories.
        """
        print("Source directory:", os.path.join( self.config['SDTDATA_SCIDESTDIR'], folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        matched_folders = []
        all_targets = os.listdir(self.config['SDTDATA_SCIDESTDIR'])
        for target in all_targets:
            if folder_key in target:
                matched_folders.append(target)
        all_matched_folders = set(matched_folders)
        return sorted(all_matched_folders)
            
    def _run_obsrsync(self, telescope_id, foldername):
        """
        Moves all FITS files from a telescope's folder into a temporary directory.

        :param telescope_id: The specific telescope folder (e.g., '7DT01', '7DT02')
        :param foldername: The folder containing FITS files.
        """
        src_folder = os.path.join(self.config['SDTDATA_OBSSOURCEDIR'], telescope_id, foldername)
        dest_folder = os.path.join(self.config['SDTDATA_OBSDESTDIR'], telescope_id, foldername)

        if not os.path.exists(src_folder):
            print(f"Source folder does not exist: {src_folder}")
            return
            
        # Ensure destination directory exists
        os.makedirs(dest_folder, exist_ok=True)

        # Rsync all files to the temporary location
        cmd = ["rsync", "-av", "--progress", src_folder + "/", dest_folder + "/"]
        print(f"Moving all files for {telescope_id} -> {dest_folder}")
        subprocess.run(cmd)

        return dest_folder

    def sync_obsdata(self, foldername: str):
        """
        Moves observational data, then sorts them into correct folders by binning and gain.

        :param foldername: The folder name containing FITS files.
        """
        # Step 1: Get telescope directories
        source_pattern = os.path.join(self.config['SDTDATA_OBSSOURCEDIR'],"7DT??", foldername)
        telescope_dirs = glob.glob(source_pattern)
        
        # Step 2: Extract telescope IDs
        telescope_ids = [os.path.basename(os.path.dirname(os.path.normpath(t))) for t in telescope_dirs]

        if not telescope_ids:
            print("No telescope folders found.")

        # Step 3: Move all files to temporary storage in parallel
        with Pool(processes=len(telescope_ids)) as pool:
            dest_folders = pool.starmap(self._run_obsrsync, [(tid, foldername) for tid in telescope_ids])


    def _run_scirsync(self, telescope_id, targetname):
        """
        Moves all FITS files from a telescope's folder into a temporary directory.

        :param telescope_id: The specific telescope folder (e.g., '7DT01', '7DT02')
        :param targetname: The folder containing FITS files.
        """
        src_folder = os.path.join(self.config['SDTDATA_SCISOURCEDIR'], targetname, telescope_id)
        dest_folder = os.path.join(self.config['SDTDATA_SCIDESTDIR'], targetname, telescope_id)

        if not os.path.exists(src_folder):
            print(f"Source folder does not exist: {src_folder}")
            return
            
        # Ensure destination directory exists
        os.makedirs(dest_folder, exist_ok=True)

        # Rsync all files to the temporary location
        cmd = ["rsync", "-av", "--progress", src_folder + "/", dest_folder + "/"]
        print(f"Moving all files for {telescope_id} -> {dest_folder}")
        subprocess.run(cmd)

        return dest_folder

    def sync_scidata(self, targetname : str):
        """
        Moves calibration data, then sorts them into correct folders by binning and gain.

        :param targetname: The target name containing FITS files.
        """
        # Step 1: Get telescope directories
        source_pattern = os.path.join(self.config['SDTDATA_SCISOURCEDIR'], targetname, "7DT??")
        telescope_dirs = glob.glob(source_pattern)
        
        # Step 2: Extract telescope IDs
        telescope_ids = [os.path.basename((os.path.normpath(t))) for t in telescope_dirs]

        if not telescope_ids:
            print("No telescope folders found.")

        # Step 3: Move all files to temporary storage in parallel
        with Pool(processes=len(telescope_ids)) as pool:
            dest_folders = pool.starmap(self._run_scirsync, [(tid, targetname) for tid in telescope_ids])
#%%
# Example usage:
if __name__ == "__main__":
    foldername = "2025"  # Add required folder keys
    self = SDTData()
    #tbl = self.show_destdata('2025-02-10_gain0')
    targetname = 'T03898'
    #self.sync_obsdata(foldername)
    self.show_obsdestdata_folder(foldername)
    #self.sync_scidata(targetname = targetname)
    #sync_manager.sync_all_folders(folder_keys)

# %%
