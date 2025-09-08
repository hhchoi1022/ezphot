#%%
from tqdm import tqdm
import inspect
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

    def __repr__(self):
        return f"SDTDataQuerier(ccd={self.ccd})\n For help, use 'help(self)' or `self.help()`."

    def help(self):
        # Get all public methods from the class, excluding `help`
        methods = [
            (name, obj)
            for name, obj in inspect.getmembers(self.__class__, inspect.isfunction)
            if not name.startswith("_") and name != "help"
        ]

        # Build plain text list with parameters
        lines = []
        for name, func in methods:
            sig = inspect.signature(func)
            params = [str(p) for p in sig.parameters.values() if p.name != "self"]
            sig_str = f"({', '.join(params)})" if params else "()"
            lines.append(f"- {name}{sig_str}")

        # Final plain text output
        help_text = ""
        print(f"Help for {self.__class__.__name__}\n{help_text}\n\nPublic methods:\n" + "\n".join(lines))
        
    def sync_obsdata(self, 
                     foldername: str,
                     file_pattern: str = '*.fits',
                     ignore_exists: bool = True):
        """
        Syncs all FITS files from a given foldername in the source directory to the destination directory.
        
        This function opens multiple rsync processes to sync the data. (n_processes = len(telescope_ids))
        The destination directory is defined in the helper.config['SDTDATA_OBSDESTDIR']
        
        Parameters
        ----------
        foldername : str
            The foldername to sync.
        file_pattern : str
            The file pattern to sync.
        ignore_exists : bool
            If True, ignore the files that already exist in the destination directory.
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
            dest_folders = pool.starmap(self._run_obsrsync, [(tid, foldername, file_pattern, ignore_exists) for tid in telescope_ids])

    def sync_scidata(self, targetname : str, 
                     file_pattern: str = '*.fits',
                     ignore_exists: bool = True):
        """
        Syncs all FITS files from a given targetname in the source directory to the destination directory.
        
        This function opens multiple rsync processes to sync the data. (n_processes = len(telescope_ids))
        The destination directory is defined in the helper.config['SDTDATA_SCIDESTDIR']
        
        Parameters
        ----------
        targetname : str
            The target name to sync.
        file_pattern : str
            The file pattern to sync.
        ignore_exists : bool
            If True, ignore the files that already exist in the destination directory.
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
            dest_folders = pool.starmap(self._run_scirsync, [(tid, targetname, file_pattern, ignore_exists) for tid in telescope_ids])

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
            Glob pattern to match FITS files (default: ``'*.fits'``).

        Returns
        -------
        fits_counts : dict
            Dictionary of {telescope_id: count or list of file paths}, sorted by telescope ID.
        """
        import glob
        import os

        fits_counts = {}

        # Find all 7DT?? telescope directories
        telescope_dirs = glob.glob(os.path.join(self.helper.config['SDTDATA_OBSSOURCEDIR'], "7DT??"))
        print(f"Found {len(telescope_dirs)} telescope directories.")

        for telescope_dir in tqdm(telescope_dirs, desc="Searching telescopes..."):
            telescope_id = os.path.basename(telescope_dir)
            folder_path = os.path.join(telescope_dir, foldername)

            if os.path.isdir(folder_path):
                fits_files = glob.glob(os.path.join(folder_path, pattern))

                if show_only_numbers:
                    fits_counts[telescope_id] = len(fits_files)
                else:
                    fits_counts[telescope_id] = sorted(fits_files)

        sorted_fits_counts = {tid: fits_counts[tid] for tid in sorted(fits_counts)}
        total_counts = sum([len(files) if isinstance(files, list) else files for files in sorted_fits_counts.values()])

        if not sorted_fits_counts:
            print("No matched folders found.")
        else:
            print(f"Total files found: {total_counts}")
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
            print("No matched folders found.")
        else:
            print(sorted_fits_counts)

        return sorted_fits_counts


    def show_obssourcefolder(self, 
                             folder_key : str = '*'):
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
            
        sorted_folders = sorted(folders)
                
        matched_folders = []
        for folder in sorted_folders:
            if folder_key in folder:
                matched_folders.append(folder)
            else:
                pass
        if not matched_folders:
            print("No matched folders found.")
        else:
            print(f"{len(matched_folders)} folders found.")
            return matched_folders

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
                
        sorted_folders = sorted(folders)
            
        matched_folders = []
        for folder in sorted_folders:
            if folder_key in folder:
                matched_folders.append(folder)
            else:
                pass
        if not matched_folders:
            print("No matched folders found.")
        else:
            print(f"{len(matched_folders)} folders found.")
            return matched_folders
        
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
        print(f'# of directories: {len(dirs)}')

        for dir in tqdm(dirs, desc="Searching directories..."):
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
        total_counts = sum([len(files) if isinstance(files, list) else files for files in sorted_fits_counts.values()])

        if not sorted_fits_counts:
            print("No matched targets found.")
            return None
        else:
            print(f"Total files found: {total_counts}")
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
        total_counts = sum([len(files) if isinstance(files, list) else files for files in sorted_fits_counts.values()])
        if not sorted_fits_counts:
            print("No matched folders found.")
            return None
        else:
            print(f"Total files found: {total_counts}")
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
        print(f"{len(all_matched_folders)} folders found.")
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

    def _run_obsrsync(self, telescope_id, foldername, file_pattern='*com.fits', ignore_exists=True):
        """
        Copy ONLY files matching file_pattern from src to dest (recursively).
        """
        src_folder  = os.path.join(self.helper.config['SDTDATA_OBSSOURCEDIR'], telescope_id, foldername)
        dest_folder = os.path.join(self.helper.config['SDTDATA_OBSDESTDIR'],   telescope_id, foldername)

        if not os.path.exists(src_folder):
            print(f"Source folder does not exist: {src_folder}")
            return

        os.makedirs(dest_folder, exist_ok=True)

        # Normalize to list (support multiple patterns if needed)
        includes = [file_pattern] if isinstance(file_pattern, str) else list(file_pattern or [])

        cmd = [
            "rsync", "-av",
            "--info=progress2",
            "--no-inc-recursive",   # lowers memory on big trees
            "--prune-empty-dirs",   # don't create empty dirs at dest
            "--ignore-existing" if ignore_exists else "",
            "--include", "*/"       # allow directory traversal
        ]
        for pat in includes:
            cmd += ["--include", pat]

        # IMPORTANT: exclude everything else
        cmd += ["--exclude", "*", src_folder + "/", dest_folder + "/"]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return dest_folder
    
    
    def _run_scirsync(self, telescope_id, targetname, file_pattern='*com.fits', ignore_exists=True):
        """
        Copy ONLY files matching file_pattern from src to dest (recursively).
        """
        src_folder  = os.path.join(self.helper.config['SDTDATA_SCISOURCEDIR'], targetname, telescope_id)
        dest_folder = os.path.join(self.helper.config['SDTDATA_SCIDESTDIR'],   targetname, telescope_id)

        if not os.path.exists(src_folder):
            print(f"Source folder does not exist: {src_folder}")
            return

        os.makedirs(dest_folder, exist_ok=True)

        # Normalize to list (support multiple patterns if needed)
        includes = [file_pattern] if isinstance(file_pattern, str) else list(file_pattern or [])

        cmd = [
            "rsync", "-av",
            "--info=progress2",
            "--no-inc-recursive",   # lowers memory on big trees
            "--prune-empty-dirs",   # don't create empty dirs at dest
            "--ignore-existing" if ignore_exists else "",
            "--include", "*/"       # allow directory traversal
        ]
        for pat in includes:
            cmd += ["--include", pat]

        # IMPORTANT: exclude everything else
        cmd += ["--exclude", "*", src_folder + "/", dest_folder + "/"]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
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
    folder_key = '*'
    #self.sync_obsdata(foldername)
    #data = self.show_scisourcedata(targetname)
    import time
    # for targetname in tile_id_list:
    #     print(f"Syncing data for target: {targetname}")
    #     self.sync_scidata(targetname = targetname)
    #     time.sleep(10)
    #sync_manager.sync_all_folders(folder_keys)
    #a = self.show_obssourcefolder(folder_key)
    #a = self.show_scisourcefolder(folder_key = '*')


#%%
# Example usage:
if __name__ == "__main__":
    foldername = "2025"  # Add required folder keys
    self = SDTDataQuerier()
    #tbl = self.show_destdata('2025-02-10_gain0')
    tile_id_list = set([
    "T22956"
    ])
    folder_key = '*'
    #self.sync_obsdata(foldername)
    #data = self.show_scisourcedata(targetname)
    import time
    # for targetname in tile_id_list:
    #     print(f"Syncing data for target: {targetname}")
    #     self.sync_scidata(targetname = targetname)
    #     time.sleep(10)
    #sync_manager.sync_all_folders(folder_keys)
    #a = self.show_obssourcefolder(folder_key)
    #a = self.show_scisourcefolder(folder_key = '*')
