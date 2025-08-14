#%%
import os
import glob
import subprocess
from multiprocessing import Pool

from ezphot.helper import Helper
#%%

class SDTDataQuerier:
    """
    SDTDataQuerier is a class that syncs data from the source directory to the destination directory.
    
    This class provides
    
    1. Syncing observational data
    
    2. Syncing calibrated data
    
    3. Showing the list of files in the source and destination directories
    
    4. Showing the list of folders in the source and destination directories
    """
    def __init__(self, 
                 ccd : str = 'C361K'):
        self.helper = Helper()
        self.folders = []
        self.ccd = ccd
        

    def sync_obsdata(self, 
                     foldername: str):
        """
        Syncs all FITS files from a given foldername in the source directory to the destination directory.
        
        This function opens multiple rsync processes to sync the data. (n_processes = len(telescope_ids))
        The destination directory is defined in the helper.config['SDTDATA_OBSDESTDIR']
        
        Parameters
        ----------
        foldername : str
            The foldername to sync.
        """
        # Step 1: Get telescope directories
        source_pattern = os.path.join(self.helper.config['SDTDATA_OBSSOURCEDIR'],"7DT??", foldername)
        telescope_dirs = glob.glob(source_pattern)
        
        # Step 2: Extract telescope IDs
        telescope_ids = [os.path.basename(os.path.dirname(os.path.normpath(t))) for t in telescope_dirs]

        if not telescope_ids:
            print("No telescope folders found.")

        # Step 3: Move all files to temporary storage in parallel
        with Pool(processes=len(telescope_ids)) as pool:
            dest_folders = pool.starmap(self._run_obsrsync, [(tid, foldername) for tid in telescope_ids])

    def sync_scidata(self, targetname : str):
        """
        Syncs all FITS files from a given targetname in the source directory to the destination directory.
        
        This function opens multiple rsync processes to sync the data. (n_processes = len(telescope_ids))
        The destination directory is defined in the helper.config['SDTDATA_SCIDESTDIR']
        
        Parameters
        ----------
        targetname : str
            The target name to sync.
        """
        # Step 1: Get telescope directories
        source_pattern = os.path.join(self.helper.config['SDTDATA_SCISOURCEDIR'], targetname, "7DT??")
        telescope_dirs = glob.glob(source_pattern)
        
        # Step 2: Extract telescope IDs
        telescope_ids = [os.path.basename((os.path.normpath(t))) for t in telescope_dirs]

        if not telescope_ids:
            print("No telescope folders found.")

        # Step 3: Move all files to temporary storage in parallel
        with Pool(processes=len(telescope_ids)) as pool:
            dest_folders = pool.starmap(self._run_scirsync, [(tid, targetname) for tid in telescope_ids])

    def show_obssourcedata(self, 
                           foldername: str, 
                           show_only_numbers: bool = False,
                           pattern: str = '*.fits'):
        """
        Shows the number or list of FITS files matching a pattern in a given folder across all 7DT?? telescope directories.

        Parameters
        ----------
        foldername : str
            Subfolder name (e.g., filter name) inside each 7DT?? directory.
        show_only_numbers : bool
            If True, return only the number of matched FITS files per telescope.
        pattern : str
            Glob pattern to match FITS files (default: '*.fits').

        Returns
        -------
        dict
            Dictionary of {telescope_id: count or list of file paths}, sorted by telescope ID.
        """
        import glob
        import os

        fits_counts = {}

        # Find all 7DT?? telescope directories
        telescope_dirs = glob.glob(os.path.join(self.config['SDTDATA_OBSSOURCEDIR'], "7DT??"))

        for telescope_dir in telescope_dirs:
            telescope_id = os.path.basename(telescope_dir)
            folder_path = os.path.join(telescope_dir, foldername)

            if os.path.isdir(folder_path):
                fits_files = glob.glob(os.path.join(folder_path, pattern))

                if show_only_numbers:
                    fits_counts[telescope_id] = len(fits_files)
                else:
                    fits_counts[telescope_id] = sorted(fits_files)

        sorted_fits_counts = {tid: fits_counts[tid] for tid in sorted(fits_counts)}

        if not sorted_fits_counts:
            print("No matching folders found.")
        else:
            print(sorted_fits_counts)

        return sorted_fits_counts

    def show_obsdestdata(self, 
                         foldername: str, 
                         show_only_numbers: bool = False,
                         pattern: str = '*.fits'):
        """
        Shows the number or list of FITS files matching a pattern in a given folder across all 7DT?? telescopes.

        Parameters
        ----------
        foldername : str
            Subfolder name (e.g., filter name) inside each 7DT?? directory.
        show_only_numbers : bool
            If True, only show counts of matched files.
        pattern : str
            Pattern to match FITS files (e.g., '*.fits', 'calib*.fits').

        Returns
        -------
        dict
            Dictionary keyed by telescope ID, containing counts or file lists.
        """
        import glob
        import os

        fits_counts = {}

        telescope_dirs = glob.glob(os.path.join(self.helper.config['SDTDATA_OBSDESTDIR'], '7DT??', foldername))

        for telescope_dir in telescope_dirs:
            telescope_id = os.path.basename(os.path.dirname(telescope_dir))
            fits_files = glob.glob(os.path.join(telescope_dir, pattern))

            if show_only_numbers:
                fits_counts[telescope_id] = len(fits_files)
            else:
                fits_counts[telescope_id] = sorted(fits_files)

        sorted_fits_counts = {tid: fits_counts[tid] for tid in sorted(fits_counts)}

        if not sorted_fits_counts:
            print("No matching folders found.")
        else:
            print(sorted_fits_counts)

        return sorted_fits_counts


    def show_obssourcefolder(self, 
                             folder_key : str = None):
        """
        Shows the contents of the source and destination directories.
        """
        print("Source directory:", os.path.join( self.helper.config['SDTDATA_OBSSOURCEDIR'], "7DT??", folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        folders = set()

        for entry in os.scandir( self.helper.config['SDTDATA_OBSSOURCEDIR']):
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

    def show_obsdestfolder(self, 
                           folder_key : str = None):
        """
        Shows the contents of the source and destination directories.
        """
        print("Source directory:", os.path.join( self.helper.config['SDTDATA_OBSDESTDIR'], "7DT??", folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        folders = set()

        for entry in os.scandir( self.helper.config['SDTDATA_OBSDESTDIR']):
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
        
    def show_scisourcedata(self, 
                           targetname: str, 
                           show_only_numbers: bool = False, 
                           key: str = 'filter',  # 'filter' or 'telescope'
                           file_pattern: str = '*.fits'  # e.g., '*.fits', 'calib*.fits'
                           ):
        """
        Shows the number of FITS files matching a pattern for each specified folder across all 7DT?? directories.

        Parameters
        ----------
        targetname : str
            Target name under SDTDATA_SCISOURCEDIR.
        show_only_numbers : bool
            If True, only show counts instead of filenames.
        key : str
            'filter' (default) to group by filter folders under telescopes, or 'telescope' to group only by telescope.
        file_pattern : str
            File pattern to match FITS files, e.g., '*.fits', 'calib*.fits', '*stack*.fits'.

        Returns
        -------
        dict
            Dictionary of telescope or filter-wise counts or file lists.
        """
        import glob
        import os

        fits_counts = {}

        if key.lower() == 'filter':
            dirs = glob.glob(os.path.join(self.helper.config['SDTDATA_SCISOURCEDIR'], targetname, "7DT??", '*'))
        elif key.lower() == 'telescope':
            dirs = glob.glob(os.path.join(self.helper.config['SDTDATA_SCISOURCEDIR'], targetname, "7DT??"))
        else:
            raise ValueError("Invalid key. Must be 'filter' or 'telescope'.")

        for dir in dirs:
            id_ = os.path.basename(dir)

            if key.lower() == 'filter':
                fits_files = glob.glob(os.path.join(dir, file_pattern))
            else:
                fits_files = glob.glob(os.path.join(dir, "*", file_pattern))

            if id_ not in fits_counts:
                fits_counts[id_] = 0 if show_only_numbers else []

            if show_only_numbers:
                fits_counts[id_] += len(fits_files)
            else:
                fits_counts[id_].extend(sorted(fits_files))

        sorted_fits_counts = {id_: fits_counts[id_] for id_ in sorted(fits_counts)}

        if not sorted_fits_counts:
            print("No matching folders found.")
        else:
            print(sorted_fits_counts)

        return sorted_fits_counts
    
    def show_scidestdata(self, 
                         targetname: str, 
                         show_only_numbers: bool = False,
                         key: str = 'filter',  # 'filter' or 'telescope'
                         pattern: str = '*.fits'  # e.g., '*.fits', 'calib*.fits'
                         ):
        """
        Shows the number of FITS files matching a pattern for each specified folder across all 7DT?? directories.

        Parameters
        ----------
        targetname : str
            Target name under SDTDATA_SCIDESTDIR.
        show_only_numbers : bool
            If True, only show counts instead of filenames.
        key : str
            'filter' (default) to group by filter folders under telescopes, or 'telescope' to group only by telescope.
        pattern : str
            File pattern to match FITS files, e.g., '*.fits', 'calib*.fits'.

        Returns
        -------
        dict
            Dictionary of telescope or filter-wise counts or file lists.
        """
        import glob
        import os

        fits_counts = {}

        if key.lower() == 'filter':
            dirs = glob.glob(os.path.join(self.helper.config['SDTDATA_SCIDESTDIR'], targetname, "7DT??", '*'))
        elif key.lower() == 'telescope':
            dirs = glob.glob(os.path.join(self.helper.config['SDTDATA_SCIDESTDIR'], targetname, "7DT??"))
        else:
            raise ValueError("Invalid key. Must be 'filter' or 'telescope'.")

        for dir in dirs:
            id_ = os.path.basename(dir)

            if key.lower() == 'filter':
                fits_files = glob.glob(os.path.join(dir, pattern))
            else:
                fits_files = glob.glob(os.path.join(dir, "*", pattern))

            if id_ not in fits_counts:
                fits_counts[id_] = 0 if show_only_numbers else []

            if show_only_numbers:
                fits_counts[id_] += len(fits_files)
            else:
                fits_counts[id_].extend(sorted(fits_files))

        sorted_fits_counts = {id_: fits_counts[id_] for id_ in sorted(fits_counts)}

        if not sorted_fits_counts:
            print("No matching folders found.")
        else:
            print(sorted_fits_counts)

        return sorted_fits_counts


    def show_scisourcefolder(self, 
                             folder_key : str = '*'):
        """
        Shows the contents of the source directory.
        
        Parameters
        ----------
        folder_key : str
            The folder key to show.
            
        Returns
        -------
        list of folder names: list
            List of folder names that match the folder key.
        """
        print("Source directory:", os.path.join( self.helper.config['SDTDATA_SCISOURCEDIR'], folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        matched_folders = []
        all_targets = os.listdir(self.helper.config['SDTDATA_SCISOURCEDIR'])
        for target in all_targets:
            if folder_key in target:
                matched_folders.append(target)
        all_matched_folders = set(matched_folders)
        return sorted(all_matched_folders)

    def show_obsdestfolder(self, 
                           folder_key : str = '*'):
        """
        Shows the contents of the source and destination directories.
        
        Parameters
        ----------
        folder_key : str
            The folder key to show.
            
        Returns
        -------
        list of folder names: list
            List of folder names that match the folder key.
        """
        print("Source directory:", os.path.join( self.helper.config['SDTDATA_SCIDESTDIR'], folder_key))
        if "*" in folder_key:
            folder_key = folder_key.replace("*", "")

        matched_folders = []
        all_targets = os.listdir(self.helper.config['SDTDATA_SCIDESTDIR'])
        for target in all_targets:
            if folder_key in target:
                matched_folders.append(target)
        all_matched_folders = set(matched_folders)
        return sorted(all_matched_folders)
            
    def _run_obsrsync(self, telescope_id, foldername):
        src_folder = os.path.join(self.helper.config['SDTDATA_OBSSOURCEDIR'], telescope_id, foldername)
        dest_folder = os.path.join(self.helper.config['SDTDATA_OBSDESTDIR'], telescope_id, foldername)

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

    def _run_scirsync(self, telescope_id, targetname):
        """
        Moves all FITS files from a telescope's folder into a temporary directory.

        :param telescope_id: The specific telescope folder (e.g., '7DT01', '7DT02')
        :param targetname: The folder containing FITS files.
        """
        src_folder = os.path.join(self.helper.config['SDTDATA_SCISOURCEDIR'], targetname, telescope_id)
        dest_folder = os.path.join(self.helper.config['SDTDATA_SCIDESTDIR'], targetname, telescope_id)

        if not os.path.exists(src_folder):
            print(f"Source folder does not exist: {src_folder}")
            return
            
        # Ensure destination directory exists
        os.makedirs(dest_folder, exist_ok=True)

        # Rsync all files to the temporary location
        #cmd = ["rsync", "-av", "--progress", "--exclude", "*.png", "--exclude", "*.cat", src_folder + "/", dest_folder + "/"]
        cmd = ["rsync", "-av", "--progress", "--exclude", "*.png", src_folder + "/", dest_folder + "/"]
        print(f"Moving all files for {telescope_id} -> {dest_folder}")
        subprocess.run(cmd)

        return dest_folder
    

#%%
# Example usage:
if __name__ == "__main__":
    foldername = "2025"  # Add required folder keys
    self = SDTDataQuerier()
    #tbl = self.show_destdata('2025-02-10_gain0')
    tile_id_list = set([
    "T22956"
    ])
    #self.sync_obsdata(foldername)
    #data = self.show_scisourcedata(targetname)
    import time
    for targetname in tile_id_list:
        print(f"Syncing data for target: {targetname}")
        self.sync_scidata(targetname = targetname)
        time.sleep(10)
    #sync_manager.sync_all_folders(folder_keys)

# %%
