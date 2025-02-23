#%%
import glob
import os
from datetime import datetime
import json
import multiprocessing 
from multiprocessing import Manager
import gc
import numpy as np
from tqdm import tqdm
import time
import inspect

from Research.helper import Helper
from Research.photometry import Image
#%%

class Photometry:
    
    def __init__(self, 
                 imagelist : list, 
                 telescope_info: dict = None,
                 reference_image : str = None):
        self.helper = Helper()
        self.reference_image = reference_image
        if telescope_info == None:
            telescope_info = self.get_telinfo()    
        self.telinfo = telescope_info
        
        #self.shared_memory_manager = Manager()
        self.original_imagelist = imagelist
        self.target_imagelist = imagelist
        self.failed_imagelist = dict()
        self.failed_imagelist['subbkg'] = list()
        self.failed_imagelist['astrometry'] = list()
        self.failed_imagelist['calculate'] = list()
        self.failed_imagelist['cutout'] = list()
        self.failed_imagelist['align'] = list()
        self.failed_imagelist['combine'] = list()
        self.failed_imagelist['subtract'] = list()
        self.failed_imagelist['photometry'] = list()

    def __repr__(self):
        methods = [f'Photometry.{name}()\n' for name, method in inspect.getmembers(
            Photometry, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt
    
    def exclude_outlier(self, 
                        sigma : float = 5,
                        depth_cut : float = None,
                        seeing_cut : float = None,
                        depth_key : str = 'DEPTH5_APER_1_HH',
                        seeing_key : str = 'SEEING_HH',
                        visualize : bool = True,
                        write_log : bool = True,
                        move : bool = False, 
                        ):
        from astropy.stats import sigma_clip
        import matplotlib.pyplot as plt
        all_tbl = self.helper.get_imginfo(filelist = self.target_imagelist, keywords = [depth_key, seeing_key])
        masked_index = (all_tbl[depth_key].mask) | (all_tbl[seeing_key].mask)
        masked_tbl = all_tbl[masked_index]
        all_tbl = all_tbl[~masked_index]
        if depth_cut is not None:
            select_index = (np.array(all_tbl[depth_key]).astype(float) ) > depth_cut
        if seeing_cut is not None:
            select_index = (np.array(all_tbl[seeing_key]).astype(float) ) < seeing_cut
        if (depth_cut is not None) & (seeing_cut is not None):
            select_index = (np.array(all_tbl[depth_key].data).astype(float)  > depth_cut) & (np.array(all_tbl[seeing_key].data).astype(float)  < seeing_cut)
        if (depth_cut is None) & (seeing_cut is None):
            select_index = ~(sigma_clip(list(all_tbl[depth_key]),sigma = sigma, maxiters= 5).mask) & ~(sigma_clip(list(all_tbl[seeing_key]),sigma = sigma, maxiters= 5).mask)
        cut_tbl = all_tbl[~select_index]
        selected_tbl = all_tbl[select_index]
        
        # Update target_imagelist
        self.failed_imagelist['outlier'] = list(cut_tbl['file'])
        self.target_imagelist = selected_tbl['file']
        
        if visualize:
            plt.figure(figsize = (10,5))
            plt.scatter(all_tbl[seeing_key],all_tbl[depth_key], c= 'k', marker = 'o', alpha = 0.6)
            plt.scatter(cut_tbl[seeing_key],cut_tbl[depth_key], c= 'r', label = 'Outlier', marker = 'o', alpha = 0.6)
            plt.xlabel('Seeing[arcsec]')
            plt.ylabel('Depth[AB]')
            plt.grid()
            plt.legend()
            plt.show()
        if write_log:
            # Save
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_outlier_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_outlier_failed_imagelist.txt")   
        if (move == True) & (len(cut_tbl) > 0):
            for image in cut_tbl['file']:
                os.makedirs(os.path.join(os.path.dirname(image), "outlier"), exist_ok=True)
                os.system(f'mv {image} {os.path.join(os.path.dirname(image), "outlier")}')
        if (move == True) & (len(masked_tbl) > 0):
            for image in masked_tbl['file']:
                os.makedirs(os.path.join(os.path.dirname(image), "masked"), exist_ok=True)
                os.system(f'mv {image} {os.path.join(os.path.dirname(image), "masked")}')
        

    def scamp(self,
              filelist : list = None,
              sex_configfile : str = None,
              scamp_configfile : str = None,
              print_output : bool = True):
        
        if filelist is None:
            filelist = list(self.target_imagelist).copy()
        
        self.helper.print(f'Start running SCAMP...', print_output)
        if sex_configfile == None:
            sex_configfile = f"{os.path.join(self.helper.sexpath,self.telinfo['obs'])}.scampconfig"
        result = self.helper.run_scamp(filelist = self.target_image,
                                       sex_configfile = sex_configfile,
                                       scamp_configfile = scamp_configfile,
                                       update_files = True,
                                       print_output = False
                                       )
        self.helper.print(f'SCAMP is finished', print_output)
    
    ################# START Astrometry #################
    def astrometry(self, 
                   filelist : list = None,
                   sex_configfile : str = None,
                   ra : float = None,
                   dec : float = None,
                   radius : float = None,
                   scalelow : float = 0.6, 
                   scalehigh : float = 0.8, 
                   overwrite : bool = False, 
                   write_log : bool = True,
                   print_output : bool = True,
                   remove : bool = True
                   ):
        
        if filelist is None:
            filelist = self.target_imagelist
        
        succeeded_imagelist = []
        failed_imagelist = {key: [] for key in self.failed_imagelist.keys()}    
        if print_output:
            iter_instance = tqdm(filelist, desc = 'Astrometry images...')
        else:
            iter_instance = filelist
        for file_ in iter_instance:
            im = Image(image=file_, telescope_info=self.telinfo, reference_image=self.reference_image)
            try:
                im.astrometry(sex_configfile = sex_configfile,
                              ra = ra,
                              dec = dec,
                              radius = radius,
                              scalelow = scalelow,
                              scalehigh = scalehigh,
                              overwrite = overwrite,
                              print_output = False,
                              remove = remove,
                              )
                succeeded_imagelist.append(im.target_image)
            except:
                failed_imagelist['astrometry'].append(file_)
        
        # Update targetlist
        self.target_imagelist = succeeded_imagelist
        self.failed_imagelist = failed_imagelist
        
        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_astrometry_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_astrometry_failed_imagelist.txt")
        
        # Remove astrometry files
        if remove:
            image_paths = set([os.path.dirname(file) for file in filelist])
            for path in image_paths:
                os.chdir(path)
                os.system(f'rm tmp* astrometry* *.conv default.nnw *.wcs *.rdls *.corr *.xyls *.solved *.axy *.match check.fits *.param {os.path.basename(sex_configfile)}')
        
        return succeeded_imagelist, failed_imagelist
 
    def faster_astrometry(self,
                          filelist : list = None,
                          sex_configfile : str = None,
                          num_processes : int = 3,
                          ra : float = None,
                          dec : float = None,
                          radius : float = None,
                          scalelow : float = 0.6, 
                          scalehigh : float = 0.8, 
                          overwrite : bool = False, 
                          write_log : bool = True,
                          print_output : bool = True,
                          remove : bool = True,
                          ):
        if filelist is None:
            filelist = self.target_imagelist
            
        # Split the target image list into groups
        self.helper.print(f"Splitting image list into {num_processes} groups...", print_output)
        num_images = len(filelist)
        num_images_per_group = num_images // num_processes
        image_groups = [filelist[i * num_images_per_group:(i + 1) * num_images_per_group] for i in range(num_processes - 1)]
        image_groups.append(filelist[(num_processes - 1) * num_images_per_group:])
        
        # Create a pool of processes and map the worker function to each group
        self.helper.print('Run astrometry with {} processes'.format(num_processes), print_output)
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for group in image_groups:
                kwargs = dict(filelist=group,
                              sex_configfile=sex_configfile,
                              ra=ra,
                              dec=dec,
                              radius=radius,
                              scalelow=scalelow,
                              scalehigh=scalehigh,
                              overwrite=overwrite,
                              write_log=False,
                              print_output=print_output,
                              remove=remove
                              )
                result = pool.apply_async(self.astrometry, kwds=kwargs)
                results.append(result)
            
            # Wait for all results to complete
            output = [result.get() for result in results]
        
        # Update target_imagelist
        self.target_imagelist = [item for sublist in output for item in sublist[0]]
        combined_dict = {}
        for dict_component in [sublist[1] for sublist in output]:
            for key, value in dict_component.items():
                if key not in combined_dict:
                    combined_dict[key] = []  # Initialize the key with an empty list if not present
                combined_dict[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = combined_dict

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastastrometry_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastastrometry_failed_imagelist.txt")

    def calculate(self, 
                  filelist: list = None,
                  sex_configfile: str = None,  # Absolute Path
                  detect_threshold: float = 3.0,
                  aperture_type: str = 'relative',  # relative or absolute
                  aperture_sizes: list = [1.5, 2.5, 3.5],  # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                  ref_catalog_name: str = 'APASS',
                  ref_catalog_conversion : str = None,
                  ref_maxmag=16,
                  ref_minmag=12,
                  visualize: bool = True,
                  update_header: bool = True,
                  check_zp_by_color: bool = False,
                  write_log : bool = True,
                  print_output : bool = True):
        
        if filelist is None:
            filelist = self.target_imagelist
        
        succeeded_imagelist = []
        failed_imagelist = {key: [] for key in self.failed_imagelist.keys()}    
        if print_output:
            iter_instance = tqdm(filelist, desc = 'Calculate images...')
        else:
            iter_instance = filelist
        for file_ in iter_instance:
            im = Image(image=file_, telescope_info=self.telinfo, reference_image=self.reference_image)
            try:
                im.calculate_zeropoint(sex_configfile=sex_configfile, 
                                       detect_threshold=detect_threshold, 
                                       aperture_type=aperture_type, 
                                       aperture_sizes=aperture_sizes, 
                                       ref_catalog_name=ref_catalog_name, 
                                       ref_catalog_conversion=ref_catalog_conversion, 
                                       ref_maxmag=ref_maxmag, 
                                       ref_minmag=ref_minmag, 
                                       visualize=visualize, 
                                       update_header=update_header,
                                       check_zp_by_color=check_zp_by_color,
                                       print_output=False)
                succeeded_imagelist.append(im.target_image)
            except:
                failed_imagelist['calculate'].append(file_)

        # Update targetlist
        self.target_imagelist = succeeded_imagelist
        self.failed_imagelist = failed_imagelist

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_calculate_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_calculate_failed_imagelist.txt")

        return succeeded_imagelist, failed_imagelist

    def faster_calculate(self, 
                         filelist : list = None,
                         sex_configfile: str = None,  # Absolute Path
                         detect_threshold: float = 3.0,
                         aperture_type: str = 'relative',  # relative or absolute
                         aperture_sizes: list = [1.5, 2.5, 3.5],  # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                         ref_catalog_name: str = 'APASS',
                         ref_catalog_conversion: str = None,
                         ref_maxmag=16,
                         ref_minmag=12,
                         visualize: bool = True,
                         update_header: bool = True,
                         check_zp_by_color: bool = False,
                         num_processes: int = 4,
                         write_log : bool = True,
                         print_output : bool = True
                         ):

        if filelist is None:
            filelist = self.target_imagelist

        # Split the target image list into groups
        self.helper.print(f"Splitting image list into {num_processes} groups...", print_output)
        num_images = len(filelist)
        num_images_per_group = num_images // num_processes
        image_groups = [filelist[i * num_images_per_group:(i + 1) * num_images_per_group] for i in range(num_processes - 1)]
        image_groups.append(filelist[(num_processes - 1) * num_images_per_group:])
        
        # Create a pool of processes and map the worker function to each group
        self.helper.print('Run calculate with {} processes'.format(num_processes), print_output)
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for group in image_groups:
                kwargs = dict(filelist=group, 
                              sex_configfile=sex_configfile, 
                              detect_threshold=detect_threshold, 
                              aperture_type=aperture_type, 
                              aperture_sizes=aperture_sizes, 
                              ref_catalog_name=ref_catalog_name, 
                              ref_catalog_conversion=ref_catalog_conversion, 
                              ref_maxmag=ref_maxmag, 
                              ref_minmag=ref_minmag, 
                              visualize=visualize, 
                              update_header=update_header,
                              check_zp_by_color=check_zp_by_color,
                              write_log = False,
                              print_output = print_output)
                result = pool.apply_async(self.calculate, kwds=kwargs)
                results.append(result)
            
            # Wait for all results to complete
            output = [result.get() for result in results]
        
        # Update target_imagelist
        self.target_imagelist = [item for sublist in output for item in sublist[0]]
        combined_dict = {}
        for dict_component in [sublist[1] for sublist in output]:
            for key, value in dict_component.items():
                if key not in combined_dict:
                    combined_dict[key] = []  # Initialize the key with an empty list if not present
                combined_dict[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = combined_dict

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastcalculate_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastcalculate_failed_imagelist.txt")

    def subtract_bkg(self, 
                     filelist: list = None,
                     apply_2D_bkg: bool = True,
                     mask_sources: bool = False,
                     mask_source_size_in_pixel : int = 10,
                     bkg_estimator: str = 'median', # mean, median, sextractor, 
                     bkg_sigma: float = 3.0, 
                     bkg_box_size: int = 300, 
                     bkg_filter_size: int = 3, 
                     prefix_subbkg : str = 'subbkg_',
                     update_header: bool = True,
                     visualize: bool = False,
                     write_log : bool = True,
                     print_output : bool = True
                     ):
        
        if filelist is None:
            filelist = self.target_imagelist
        
        succeeded_imagelist = []
        failed_imagelist = {key: [] for key in self.failed_imagelist.keys()}    
        if print_output:
            iter_instance = tqdm(filelist, desc = 'Subtracting background...')
        else:
            iter_instance = filelist
        for file_ in filelist:
            im = Image(image=file_, telescope_info=self.telinfo, reference_image=self.reference_image)
            try:
                im.subtract_bkg(apply_2D_bkg = apply_2D_bkg,
                                mask_sources = mask_sources,
                                mask_source_size_in_pixel = mask_source_size_in_pixel,
                                bkg_estimator = bkg_estimator,
                                bkg_sigma = bkg_sigma, 
                                bkg_box_size = bkg_box_size,
                                bkg_filter_size = bkg_filter_size,
                                prefix = prefix_subbkg,
                                update_header = update_header,
                                visualize = visualize,
                                print_output = print_output)
                succeeded_imagelist.append(im.target_image)
            except:
                failed_imagelist['subbkg'].append(file_)

        # Update targetlist
        self.target_imagelist = succeeded_imagelist
        self.failed_imagelist = failed_imagelist

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_subbkg_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_subbkg_failed_imagelist.txt")

        return succeeded_imagelist, failed_imagelist
    
    def faster_subtract_bkg(self,
                            num_processes : int = 3,
                            filelist: list = None,
                            apply_2D_bkg: bool = True,
                            mask_sources: bool = False,
                            mask_source_size_in_pixel : int = 10,
                            bkg_estimator: str = 'median', # mean, median, sextractor, 
                            bkg_sigma: float = 3.0, 
                            bkg_box_size: int = 300, 
                            bkg_filter_size: int = 3, 
                            prefix_subbkg : str = 'subbkg_',
                            update_header: bool = True,
                            visualize: bool = False,
                            write_log : bool = True,
                            print_output : bool = True):

        if filelist is None:
            filelist = self.target_imagelist
            
        # Split the target image list into groups
        self.helper.print(f"Splitting image list into {num_processes} groups...", print_output)
        num_images = len(filelist)
        num_images_per_group = num_images // num_processes
        image_groups = [filelist[i * num_images_per_group:(i + 1) * num_images_per_group] for i in range(num_processes - 1)]
        image_groups.append(filelist[(num_processes - 1) * num_images_per_group:])

        # Create a pool of processes and map the worker function to each group
        self.helper.print('Run bkg subtraction with {} processes'.format(num_processes), print_output)
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for group in image_groups:
                kwargs = dict(filelist=group, 
                              apply_2D_bkg = apply_2D_bkg,
                              mask_sources = mask_sources,
                              mask_source_size_in_pixel = mask_source_size_in_pixel,
                              bkg_estimator = bkg_estimator,
                              bkg_sigma = bkg_sigma,
                              bkg_box_size = bkg_box_size,
                              bkg_filter_size = bkg_filter_size,
                              prefix_subbkg = prefix_subbkg,
                              update_header = update_header,
                              visualize = visualize,
                              write_log = False,
                              print_output = print_output
                              )
                result = pool.apply_async(self.subtract_bkg, kwds=kwargs)
                results.append(result)
            
            # Wait for all results to complete
            output = [result.get() for result in results]
        
        # Update target_imagelist
        self.target_imagelist = [item for sublist in output for item in sublist[0]]
        combined_dict = {}
        for dict_component in [sublist[1] for sublist in output]:
            for key, value in dict_component.items():
                if key not in combined_dict:
                    combined_dict[key] = []  # Initialize the key with an empty list if not present
                combined_dict[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = combined_dict 
        
        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_subbkg_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_subbkg_failed_imagelist.txt")

    def cutout(self,
               filelist : list = None,
               cutout_size : int = 2000,
               write_log : bool = True,
               print_output : bool = True):
        if filelist is None:
            filelist = self.target_imagelist
        
        succeeded_imagelist = []
        failed_imagelist = {key: [] for key in self.failed_imagelist.keys()}    
        if print_output:
            iter_instance = tqdm(filelist, desc = 'Astrometry images...')
        else:
            iter_instance = filelist
        for file_ in iter_instance:
            im = Image(image=file_, telescope_info=self.telinfo, reference_image=self.reference_image)
            try:
                im.cutout(cutout_size = cutout_size)
                succeeded_imagelist.append(im.target_image)
            except:
                failed_imagelist['cutout'].append(file_)
        
        # Update targetlist
        self.target_imagelist = succeeded_imagelist
        self.failed_imagelist = failed_imagelist

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_cutout_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_cutout_failed_imagelist.txt")
        
        return succeeded_imagelist, failed_imagelist
    
    def align(self, 
              filelist : list = None,
              cut_outer : bool = False,
              outer_size : float = 0.95,
              detection_sigma : float = 5,
              write_log : bool = True,
              print_output : bool = True):
        
        if filelist is None:
            filelist = self.target_imagelist
        
        succeeded_imagelist = []
        failed_imagelist = {key: [] for key in self.failed_imagelist.keys()}    
        if print_output:
            iter_instance = tqdm(filelist, desc = 'Astrometry images...')
        else:
            iter_instance = filelist
        for file_ in iter_instance:
            im = Image(image=file_, telescope_info=self.telinfo, reference_image=self.reference_image)
            try:
                im.align(cut_outer = cut_outer, outer_size = outer_size, detection_sigma = detection_sigma, print_output = False)
            except:
                failed_imagelist['align'].append(file_)

        # Update targetlist
        self.target_imagelist = succeeded_imagelist
        self.failed_imagelist = failed_imagelist

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_align_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_align_failed_imagelist.txt")
        
        return succeeded_imagelist, failed_imagelist
        
    def faster_align(self,
                     filelist : list = None,
                     num_processes : int = 4,
                     cut_outer : bool = False,
                     outer_size : float = 0.95,
                     detection_sigma : float = 5,
                     write_log : bool = True,
                     print_output : bool = True
                     ):
        
        if filelist is None:
            filelist = self.target_imagelist

        # Split the target image list into groups
        self.helper.print(f"Splitting image list into {num_processes} groups...", print_output)
        num_images = len(self.target_imagelist)
        num_images_per_group = num_images // num_processes
        image_groups = [filelist[i * num_images_per_group:(i + 1) * num_images_per_group] for i in range(num_processes - 1)]
        image_groups.append(filelist[(num_processes - 1) * num_images_per_group:])
        
        # Create a pool of processes and map the worker function to each group
        self.helper.print('Image alignment with {} processes'.format(num_processes), print_output)
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for group in image_groups:
                kwargs = dict(filelist = group,
                              cut_outer = cut_outer,
                              outer_size = outer_size,
                              detection_sigma = detection_sigma,
                              write_log = False,
                              print_output = print_output)
                result = pool.apply_async(self.align, kwds=kwargs)
                results.append(result)
            
            # Wait for all results to complete
            output = [result.get() for result in results]
        
        # Update target_imagelist
        self.target_imagelist = [item for sublist in output for item in sublist[0]]
        combined_dict = {}
        for dict_component in [sublist[1] for sublist in output]:
            for key, value in dict_component.items():
                if key not in combined_dict:
                    combined_dict[key] = []  # Initialize the key with an empty list if not present
                combined_dict[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = combined_dict
        
        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastalign_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastalign_failed_imagelist.txt")                                                 

    def combine(self,
                filelist: list = None,
                group_key: str = 'jd',
                group_tolerance: float = 0.5,
                write_log : bool = True,
                print_output: bool = True,
                
                # Combine params
                combine_method: str = 'median',
                scale: str = 'multiply',
                prefix_combine: str = 'com_',
                zp_key: str ='ZP5_1',
                clip: str = 'extrema',
                clip_sigma_low: int = 2,
                clip_sigma_high: int = 5,
                clip_minmax_min: int = 3,
                clip_minmax_max: int = 3,
                clip_extrema_nlow: int = 1,
                clip_extrema_nhigh: int = 1,
                
                # Background Subtraction params
                subbkg_before_combine : bool = True,
                apply_2D_bkg: bool = True,
                mask_sources: bool = False,
                mask_source_size_in_pixel : int = 10,
                bkg_estimator: str = 'median', # mean, median, sextractor, 
                bkg_sigma: float = 3.0, 
                bkg_box_size: int = 300, 
                bkg_filter_size: int = 3, 
                prefix_subbkg : str = 'subbkg_',
                update_header_subbkg: bool = True,
                visualize_subbkg: bool = False,

                # Calculate params
                calculate_before_combine : bool = True,
                sex_configfile : str = None, 
                detect_threshold: float = 3.0,
                aperture_type: str = 'relative',  # relative or absolute
                aperture_sizes: list = [1.5, 2.5, 3.5],  # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                ref_catalog_name: str = 'APASS',
                ref_catalog_conversion : str = None,
                ref_maxmag : float = 16.0,
                ref_minmag : float = 12.0, 
                visualize_calculate: bool = True, 
                update_header_calculate: bool = True,
                check_zp_by_color: bool = False,
                  
                # Align params
                align_before_combine : bool = True, 
                align_cut_outer : bool = False,
                align_outer_size : float = 0.95,
                align_detection_sigma : float = 5
                ):
        
        if filelist is None:
            filelist = self.target_imagelist

        self.helper.print(f'Querying image information...', print_output)
        group_tbl = self.helper.group_table(tbl=self.helper.get_imginfo(filelist=filelist), key=group_key, tolerance=group_tolerance)
        tbl_groups = group_tbl.group_by('group').groups
        
        self.helper.print(f'From {len(filelist)} images, {len(tbl_groups)} groups are found', print_output)
        if print_output:
            iter_instance = tqdm(tbl_groups, desc = 'Image combining...')
        else:
            iter_instance = tbl_groups
            
        succeeded_imagelist = []
        failed_imagelist = {key: [] for key in self.failed_imagelist.keys()}    
        for tbl_group in iter_instance:
            files_to_combine = list()
            tbl_group.sort('jd')
            for file_ in tbl_group['file']:
                im = Image(image=file_, telescope_info=self.telinfo, reference_image=tbl_group['file'][len(tbl_group['file']) // 2])
                failed = False
                if zp_key not in im.header.keys():
                    failed = True
                
                if subbkg_before_combine and not failed:
                    try:
                        im.subtract_bkg(apply_2D_bkg = apply_2D_bkg,
                                        mask_sources = mask_sources,
                                        mask_source_size_in_pixel = mask_source_size_in_pixel,
                                        bkg_estimator = bkg_estimator,
                                        bkg_sigma = bkg_sigma, 
                                        bkg_box_size = bkg_box_size,
                                        bkg_filter_size = bkg_filter_size,
                                        prefix = prefix_subbkg,
                                        update_header = update_header_subbkg,
                                        visualize = visualize_subbkg,
                                        print_output = False)
                    except:
                        failed_imagelist['subbkg'].append(im.target_image)
                        failed = True
                    
                if calculate_before_combine and not failed:
                    try:
                        im.calculate_zeropoint(sex_configfile=sex_configfile, 
                                               detect_threshold=detect_threshold, 
                                               aperture_type=aperture_type, 
                                               aperture_sizes=aperture_sizes, 
                                               ref_catalog_name=ref_catalog_name, 
                                               ref_catalog_conversion=ref_catalog_conversion, 
                                               ref_maxmag=ref_maxmag, 
                                               ref_minmag=ref_minmag, 
                                               visualize=visualize_calculate, 
                                               update_header=update_header_calculate,
                                               check_zp_by_color=check_zp_by_color,
                                               print_output=False)
                    except:
                        failed_imagelist['calculate'].append(im.target_image)
                        failed = True
                
                if align_before_combine and not failed:
                    try:
                        im.align(cut_outer = align_cut_outer, 
                                 outer_size = align_outer_size, 
                                 detection_sigma = align_detection_sigma, 
                                 print_output = False)
                    except:
                        failed_imagelist['align'].append(im.target_image)
                        failed = True                
                
                if not failed:
                    files_to_combine.append(im.target_image)                    
                        
            if len(files_to_combine) > 0:
                try:
                    self.helper.print(f'Combining {len(files_to_combine)} images in group {tbl_group["group"][0]}', print_output)
                    combined_image = self.helper.combine_img(files_to_combine,
                                                      combine_method=combine_method,
                                                      scale=scale,
                                                      prefix=prefix_combine,
                                                      zp_key=zp_key,
                                                      print_output= print_output,
  
                                                      # Clipping
                                                      clip=clip,
                                                      clip_sigma_low=clip_sigma_low,
                                                      clip_sigma_high=clip_sigma_high,
                                                      clip_minmax_min=clip_minmax_min,
                                                      clip_minmax_max=clip_minmax_max,
                                                      clip_extrema_nlow=clip_extrema_nlow,
                                                      clip_extrema_nhigh=clip_extrema_nhigh,
                                                      )
                    succeeded_imagelist.append(combined_image)
                except:
                    failed_imagelist['combine'].extend(files_to_combine)

        # Update target_imagelist
        self.target_imagelist = succeeded_imagelist
        self.failed_imagelist = failed_imagelist

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_combine_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_combine_failed_imagelist.txt")
        
        return succeeded_imagelist, failed_imagelist

    def faster_combine(self, 
                       filelist : list = None,
                       num_processes : int = 4,
                       group_key: str = 'jd',
                       group_tolerance: float = 0.5,
                       write_log : bool = True,
                       print_output: bool = True,
                       
                       # Combine params
                       combine_method: str = 'median',
                       scale: str = 'multiply',
                       prefix_combine: str = 'com_',
                       zp_key: str ='ZP_APER_1_HH',
                       clip: str = 'extrema',
                       clip_sigma_low: int = 2,
                       clip_sigma_high: int = 5,
                       clip_minmax_min: int = 3,
                       clip_minmax_max: int = 3,
                       clip_extrema_nlow: int = 1,
                       clip_extrema_nhigh: int = 1,
                       
                       # Background Subtraction params
                       subbkg_before_combine : bool = True,
                       apply_2D_bkg: bool = True,
                       mask_sources: bool = False,
                       mask_source_size_in_pixel : int = 10,
                       bkg_estimator: str = 'median', # mean, median, sextractor, 
                       bkg_sigma: float = 3.0, 
                       bkg_box_size: int = 300, 
                       bkg_filter_size: int = 3, 
                       prefix_subbkg : str = 'subbkg_',
                       update_header_subbkg: bool = True,
                       visualize_subbkg: bool = False,
                       
                       # Calculate params
                       calculate_before_combine : bool = True,
                       sex_configfile : str = None, 
                       detect_threshold: float = 3.0,
                       aperture_type: str = 'relative',  # relative or absolute
                       aperture_sizes: list = [1.5, 2.5, 3.5],  # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                       ref_catalog_name: str = 'APASS',
                       ref_catalog_conversion : str = None,
                       ref_maxmag : float = 16.0,
                       ref_minmag : float = 12.0, 
                       visualize_calculate: bool = True, 
                       update_header_calculate: bool = True,
                       check_zp_by_color: bool = False,
                       
                       # Align params
                       align_before_combine : bool = False, 
                       align_cut_outer : bool = False,
                       align_outer_size : float = 0.95,
                       align_detection_sigma : float = 5
                       ):
        
        if filelist is None:
            filelist = self.target_imagelist

        self.helper.print(f'Querying image information...', print_output)
        groupped_tbl = self.helper.group_table(tbl=self.helper.get_imginfo(filelist=filelist), key=group_key, tolerance=group_tolerance).group_by('group')
        tbl_groups = groupped_tbl.groups
        
        num_images = len(filelist)
        num_images_per_group = num_images // num_processes
        groups_idx = [int(groupped_tbl['group'][i * num_images_per_group]) for i in range(num_processes)]
        
        image_groups = [list(tbl_groups[groups_idx[i]:groups_idx[i+1]]['file']) for i in range(len(groups_idx) - 1)]
        image_groups.append(list(tbl_groups[groups_idx[-1]:]['file']))
                
        # Create a pool of processes and map the worker function to each group
        self.helper.print('Combining images with {} processes'.format(num_processes), print_output)
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for group in image_groups:
                kwargs = dict(filelist = group,
                              group_key = group_key, 
                              group_tolerance = group_tolerance, 
                              zp_key = zp_key,
                              write_log = False,
                              print_output = print_output,
                              combine_method = combine_method,
                              scale = scale,
                              prefix_combine = prefix_combine,
                              clip = clip,
                              clip_sigma_low = clip_sigma_low,
                              clip_sigma_high = clip_sigma_high,
                              clip_minmax_min = clip_minmax_min,
                              clip_minmax_max = clip_minmax_max,
                              clip_extrema_nlow = clip_extrema_nlow,
                              clip_extrema_nhigh = clip_extrema_nhigh,
                              subbkg_before_combine = subbkg_before_combine,
                              apply_2D_bkg = apply_2D_bkg,
                              mask_sources = mask_sources,
                              mask_source_size_in_pixel = mask_source_size_in_pixel,
                              bkg_estimator = bkg_estimator,
                              bkg_sigma = bkg_sigma,
                              bkg_box_size = bkg_box_size,
                              bkg_filter_size = bkg_filter_size,
                              prefix_subbkg = prefix_subbkg,
                              update_header_subbkg = update_header_subbkg,
                              visualize_subbkg = visualize_subbkg,
                              calculate_before_combine = calculate_before_combine,
                              sex_configfile=sex_configfile, 
                              detect_threshold=detect_threshold, 
                              aperture_type=aperture_type, 
                              aperture_sizes=aperture_sizes, 
                              ref_catalog_name=ref_catalog_name, 
                              ref_catalog_conversion=ref_catalog_conversion, 
                              ref_maxmag=ref_maxmag, 
                              ref_minmag=ref_minmag, 
                              visualize_calculate=visualize_calculate, 
                              update_header_calculate=update_header_calculate,
                              check_zp_by_color=check_zp_by_color,
                              align_before_combine = align_before_combine,
                              align_cut_outer = align_cut_outer,
                              align_outer_size = align_outer_size,
                              align_detection_sigma = align_detection_sigma,
                              )
                result = pool.apply_async(self.combine, kwds=kwargs)
                results.append(result)
            
            # Wait for all results to complete
            output = [result.get() for result in results]
        
        # Update target_imagelist
        self.target_imagelist = [item for sublist in output for item in sublist[0]]
        combined_dict = {}
        for dict_component in [sublist[1] for sublist in output]:
            for key, value in dict_component.items():
                if key not in combined_dict:
                    combined_dict[key] = []  # Initialize the key with an empty list if not present
                combined_dict[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = combined_dict 
        
        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastcombine_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastcombine_imagelist.txt")

    def subtract(self,
                 filelist : list = None,
                 align_before_subtract : bool = True,
                 cutout_target_image : bool = True,
                 cutout_reference_image : bool = True,
                 cutout_size : int = 1500,
                 write_log : bool = True,
                 print_output : bool = True
                 ):
        if filelist is None:
            filelist = self.target_imagelist
        
        succeeded_imagelist = []
        failed_imagelist = {key: [] for key in self.failed_imagelist.keys()}    
        if print_output:
            iter_instance = tqdm(filelist, desc = 'Subtract images...')
        else:
            iter_instance = filelist
        for file_ in iter_instance:
            im = Image(image=file_, telescope_info=self.telinfo, reference_image=self.reference_image)
            try:
                im.subtract(align = align_before_subtract,
                            cutout_target_image = cutout_target_image,
                            cutout_reference_image = cutout_reference_image,
                            cutout_size = cutout_size, 
                            print_output = False)
                succeeded_imagelist.append(im.target_image)
            except:
                failed_imagelist['subtract'].append(file_)  
        
        # Update targetlist
        self.target_imagelist = succeeded_imagelist
        self.failed_imagelist = failed_imagelist

        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_subtract_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_subtract_failed_imagelist.txt")

        return succeeded_imagelist, failed_imagelist

    def faster_subtract(self,
                        filelist : list = None,
                        num_processes : int = 4,
                        align_before_subtract : bool = True,
                        cutout_target_image : bool = True,
                        cutout_reference_image : bool = True,
                        cutout_size : int = 1500,
                        write_log : bool = True,
                        print_output : bool = True
                        ):
        if filelist is None:
            filelist = self.target_imagelist

        # Split the target image list into groups
        self.helper.print(f"Splitting image list into {num_processes} groups...", print_output)
        num_images = len(filelist)
        num_images_per_group = num_images // num_processes
        image_groups = [filelist[i * num_images_per_group:(i + 1) * num_images_per_group] for i in range(num_processes - 1)]
        image_groups.append(filelist[(num_processes - 1) * num_images_per_group:])
        
        # Create a pool of processes and map the worker function to each group
        self.helper.print('Run subtract with {} processes'.format(num_processes), print_output)
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for group in image_groups:
                kwargs = dict(filelist=group, 
                              align_before_subtract=align_before_subtract,
                              cutout_target_image=cutout_target_image,
                              cutout_reference_image=cutout_reference_image,
                              cutout_size=cutout_size,
                              write_log = False,
                              print_output = print_output)
                result = pool.apply_async(self.subtract, kwds=kwargs)
                results.append(result)

            # Wait for all results to complete
            output = [result.get() for result in results]
        
        # Update target_imagelist
        self.target_imagelist = [item for sublist in output for item in sublist[0]]
        combined_dict = {}
        for dict_component in [sublist[1] for sublist in output]:
            for key, value in dict_component.items():
                if key not in combined_dict:
                    combined_dict[key] = []  # Initialize the key with an empty list if not present
                combined_dict[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = combined_dict
        
        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastsubtract_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_fastsubtract_failed_imagelist.txt")

    def photometry(self,                  
                   # Photometry params  
                   ra, 
                   dec, 
                   filelist : list = None,
                   sex_configfile: str = None,
                   detect_threshold : float  = 3.0,
                   aperture_type : str = 'relative', # relative or absolute
                   aperture_sizes : list = [1.5, 2.5, 3.5], # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                   visualize_photometry: bool = True,
                   write_log : bool = True,
                   print_output: bool = True,
                   
                   # Combine params
                   combine_before_photometry: bool = True,
                   group_key: str = 'jd',
                   group_tolerance: float = 0.5,
                   zp_key: str = 'ZP_APER_1_HH',
                   combine_method: str = 'median',
                   scale: str = 'multiply',
                   prefix_combine: str = 'com_',
                   clip: str = 'extrema',
                   clip_sigma_low: int = 2,
                   clip_sigma_high: int = 5,
                   clip_minmax_min: int = 3,
                   clip_minmax_max: int = 3,
                   clip_extrema_nlow: int = 1,
                   clip_extrema_nhigh: int = 1,
                   
                   # Calculate params
                   calculate_before_photometry: bool = True,
                   calculate_before_combine: bool = True,
                   ref_catalog_name: str = 'APASS',
                   ref_catalog_conversion : str = 'SDSS',
                   ref_maxmag=16,
                   ref_minmag=12,
                   visualize_calculate: bool = True,
                   check_zp_by_color : bool = False,
                   update_header_calculate : bool = True,
                   
                   # Background Subtraction params
                   subbkg_before_combine : bool = True,
                   apply_2D_bkg: bool = True,
                   mask_sources: bool = False,
                   mask_source_size_in_pixel : int = 10,
                   bkg_estimator: str = 'median', # mean, median, sextractor, 
                   bkg_sigma: float = 3.0, 
                   bkg_box_size: int = 300, 
                   bkg_filter_size: int = 3, 
                   prefix_subbkg : str = 'subbkg_',
                   update_header_subbkg: bool = True,
                   visualize_subbkg: bool = False,
                   
                   # Align params
                   align_before_combine : bool = False, 
                   align_cut_outer : bool = False,
                   align_outer_size : float = 0.95,
                   align_detection_sigma : float = 5,
                        
                   # Subtraction params
                   subtract_before_photometry : bool = True,
                   align_before_subtract : bool = True,
                   cutout_target_image : bool = True,
                   cutout_reference_image : bool = True,
                   cutout_size : int = 1500,
                   ):
        
        if filelist is None:
            filelist = self.target_imagelist
        failed_imagelist_combine = {key: [] for key in self.failed_imagelist.keys()}   
        if combine_before_photometry:
            succeeded_imagelist, failed_imagelist_combine = self.combine(filelist = filelist,
                                                                         group_key = group_key,
                                                                         group_tolerance = group_tolerance,
                                                                         write_log = False,
                                                                         print_output = print_output,
                                                                         combine_method = combine_method,
                                                                         scale = scale,
                                                                         prefix_combine = prefix_combine,
                                                                         zp_key = zp_key,
                                                                         clip = clip,
                                                                         clip_sigma_low = clip_sigma_low,
                                                                         clip_sigma_high = clip_sigma_high,
                                                                         clip_minmax_min = clip_minmax_min,
                                                                         clip_minmax_max = clip_minmax_max,
                                                                         clip_extrema_nlow = clip_extrema_nlow,
                                                                         clip_extrema_nhigh = clip_extrema_nhigh,
                                                                         subbkg_before_combine = subbkg_before_combine,
                                                                         apply_2D_bkg = apply_2D_bkg,
                                                                         mask_sources = mask_sources,
                                                                         mask_source_size_in_pixel = mask_source_size_in_pixel,
                                                                         bkg_estimator = bkg_estimator,
                                                                         bkg_sigma = bkg_sigma,
                                                                         bkg_box_size = bkg_box_size,
                                                                         bkg_filter_size = bkg_filter_size,
                                                                         prefix_subbkg = prefix_subbkg,
                                                                         update_header_subbkg = update_header_subbkg,
                                                                         visualize_subbkg = visualize_subbkg,
                                                                         calculate_before_combine = calculate_before_combine,
                                                                         sex_configfile=sex_configfile, 
                                                                         detect_threshold=detect_threshold, 
                                                                         aperture_type=aperture_type, 
                                                                         aperture_sizes=aperture_sizes, 
                                                                         ref_catalog_name=ref_catalog_name, 
                                                                         ref_catalog_conversion=ref_catalog_conversion, 
                                                                         ref_maxmag=ref_maxmag, 
                                                                         ref_minmag=ref_minmag, 
                                                                         visualize_calculate=visualize_calculate, 
                                                                         update_header_calculate=update_header_calculate,
                                                                         align_before_combine = align_before_combine,
                                                                         align_cut_outer = align_cut_outer,
                                                                         align_outer_size = align_outer_size,
                                                                         align_detection_sigma = align_detection_sigma
                                                                         )
            # Update target_imagelist
            self.target_imagelist = succeeded_imagelist
            filelist = self.target_imagelist
            
        failed_imagelist_calculate = {key: [] for key in self.failed_imagelist.keys()}   
        if combine_before_photometry or calculate_before_photometry:
            # Calcuclate zeropoint of the images
            succeeded_imagelist, failed_imagelist_calculate = self.calculate(filelist = filelist,
                                                                             sex_configfile = sex_configfile,  # Absolute Path
                                                                             detect_threshold = detect_threshold,
                                                                             aperture_type = aperture_type,  # relative or absolute
                                                                             aperture_sizes = aperture_sizes,  # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                                                                             ref_catalog_name = ref_catalog_name,
                                                                             ref_catalog_conversion = ref_catalog_conversion,
                                                                             ref_maxmag = ref_maxmag,
                                                                             ref_minmag = ref_minmag,
                                                                             visualize = visualize_calculate,
                                                                             update_header = True,
                                                                             check_zp_by_color = check_zp_by_color,
                                                                             write_log = False,
                                                                             print_output = print_output
                                                                             )
            # Update target_imagelist
            self.target_imagelist = succeeded_imagelist
            filelist = self.target_imagelist        
        
        if print_output:
            iter_instance = tqdm(filelist, desc = 'Photometry images...')
        else:
            iter_instance = filelist
        
        succeeded_imagelist = []
        failed_imagelist_photometry = {key: [] for key in self.failed_imagelist.keys()}    
        for file_ in iter_instance:
            im = Image(image=file_, telescope_info=self.telinfo, reference_image=self.reference_image)
            failed = False
            if zp_key not in im.header.keys():
                failed = True
            
            if subtract_before_photometry and not failed:
                try:
                    im.subtract(align = align_before_subtract,
                                cutout_target_image = cutout_target_image,
                                cutout_reference_image = cutout_reference_image,
                                cutout_size = cutout_size)
                except:
                    failed_imagelist_photometry['subtract'].append(im.target_image)
                    failed = True
            
            if not failed:
                try:
                    im.photometry(ra = ra,
                                  dec = dec,
                                  sex_configfile = sex_configfile, 
                                  detect_threshold = detect_threshold,
                                  aperture_type = aperture_type,
                                  aperture_sizes = aperture_sizes,
                                  visualize = visualize_photometry)
                    
                except:
                    failed_imagelist_photometry['photometry'].append(im.target_image)
                    failed = True
            if not failed:
                succeeded_imagelist.append(im.target_image)
               
        # Update target_imagelist
        self.target_imagelist = succeeded_imagelist
                
        failed_imagelist_all = {}
        for dict_component in [failed_imagelist_combine, failed_imagelist_calculate, failed_imagelist_photometry]:
            for key, value in dict_component.items():
                if key not in failed_imagelist_all:
                    failed_imagelist_all[key] = []  # Initialize the key with an empty list if not present
                failed_imagelist_all[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = failed_imagelist_all
         
        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_photometry_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_photometry_failed_imagelist.txt")
        
        return succeeded_imagelist, failed_imagelist_all

    def faster_photometry(self, 
                          # Photometry params  
                          ra, 
                          dec, 
                          num_processes : int = 4,
                          filelist : list = None,
                          sex_configfile: str = None,
                          detect_threshold : float  = 3.0,
                          aperture_type : str = 'relative', # relative or absolute
                          aperture_sizes : list = [1.5, 2.5, 3.5], # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                          visualize_photometry : bool = True,
                          write_log = True,
                          print_output: bool = True,
                          
                            # Combine params
                          combine_before_photometry: bool = True,
                          group_key: str = 'jd',
                          group_tolerance: float = 0.5,
                            zp_key: str = 'ZP_APER_1_HH',
                          combine_method: str = 'median',
                          scale: str = 'multiply',
                          prefix_combine: str = 'com_',
                          clip: str = 'extrema',
                          clip_sigma_low: int = 2,
                          clip_sigma_high: int = 5,
                          clip_minmax_min: int = 3,
                          clip_minmax_max: int = 3,
                          clip_extrema_nlow: int = 1,
                          clip_extrema_nhigh: int = 1,
                        
                          # Calculate params
                          calculate_before_photometry: bool = True,
                          calculate_before_combine : bool = True,
                          ref_catalog_name: str = 'APASS',
                          ref_catalog_conversion : str = 'SDSS',
                          ref_maxmag=16,
                          ref_minmag=12,
                          visualize_calculate: bool = True,
                          check_zp_by_color : bool = False,
                          update_header_calculate : bool = True,
                          
                          # Background Subtraction params
                          subbkg_before_combine : bool = True,
                          apply_2D_bkg: bool = True,
                          mask_sources: bool = False,
                          mask_source_size_in_pixel : int = 10,
                          bkg_estimator: str = 'median', # mean, median, sextractor, 
                          bkg_sigma: float = 3.0, 
                          bkg_box_size: int = 300, 
                          bkg_filter_size: int = 3, 
                          prefix_subbkg : str = 'subbkg_',
                          update_header_subbkg: bool = True,
                          visualize_subbkg: bool = False,
                        
                          # Align params
                          align_before_combine : bool = False, 
                          align_cut_outer : bool = False,
                          align_outer_size : float = 0.95,
                          align_detection_sigma : float = 5,
                                
                          # Subtraction params
                          subtract_before_photometry : bool = True,
                          align_before_subtract : bool = True,
                          cutout_target_image : bool = True,
                          cutout_reference_image : bool = True,
                          cutout_size : int = 1500,
                          ):

        if filelist is None:
            filelist = self.target_imagelist

        # Split the target image list into groups
        self.helper.print(f"Splitting image list into {num_processes} groups...", print_output)

        groupped_tbl = self.helper.group_table(tbl=self.helper.get_imginfo(filelist=filelist), key=group_key, tolerance=group_tolerance).group_by('group')
        tbl_groups = groupped_tbl.groups
        
        num_images = len(filelist)
        num_images_per_group = num_images // num_processes
        groups_idx = [int(groupped_tbl['group'][i * num_images_per_group]) for i in range(num_processes)]
        
        image_groups = [list(tbl_groups[groups_idx[i]:groups_idx[i+1]]['file']) for i in range(len(groups_idx) - 1)]
        image_groups.append(list(tbl_groups[groups_idx[-1]:]['file']))

        # Create a pool of processes and map the worker function to each group
        self.helper.print(f'Running photometry with {num_processes} processes', print_output)
        with multiprocessing.Pool(num_processes) as pool:
            results = []
            for group in image_groups:
                kwargs = dict(ra = ra, 
                              dec = dec, 
                              filelist = group,
                              sex_configfile = sex_configfile,
                              detect_threshold  = detect_threshold,
                              aperture_type = aperture_type,
                              aperture_sizes = aperture_sizes,
                              visualize_photometry = visualize_photometry,
                              write_log = False,
                              print_output = print_output,
                            
                              # Combine params
                              combine_before_photometry = combine_before_photometry,
                              group_key = group_key,
                              group_tolerance = group_tolerance,
                              zp_key = zp_key,
                              combine_method = combine_method,
                              scale = scale,
                              prefix_combine = prefix_combine,
                              clip = clip,
                              clip_sigma_low = clip_sigma_low,
                              clip_sigma_high = clip_sigma_high,
                              clip_minmax_min = clip_minmax_min,
                              clip_minmax_max = clip_minmax_max,
                              clip_extrema_nlow = clip_extrema_nlow,
                              clip_extrema_nhigh = clip_extrema_nhigh,
                            
                              # Calculate params
                              calculate_before_photometry = calculate_before_photometry,
                              calculate_before_combine = calculate_before_combine,
                              ref_catalog_name = ref_catalog_name,
                              ref_catalog_conversion = ref_catalog_conversion,
                              ref_maxmag= ref_maxmag,
                              ref_minmag= ref_minmag,
                              visualize_calculate = visualize_calculate,
                              check_zp_by_color = check_zp_by_color,
                              update_header_calculate = update_header_calculate,
                              
                              # Background Subtraction params
                              subbkg_before_combine = subbkg_before_combine,
                              apply_2D_bkg = apply_2D_bkg,
                              mask_sources = mask_sources,
                              mask_source_size_in_pixel = mask_source_size_in_pixel,
                              bkg_estimator = bkg_estimator, # mean, median, sextractor, 
                              bkg_sigma = bkg_sigma, 
                              bkg_box_size = bkg_box_size, 
                              bkg_filter_size = bkg_filter_size, 
                              prefix_subbkg = prefix_subbkg,
                              update_header_subbkg = update_header_subbkg,
                              visualize_subbkg = visualize_subbkg,
                            
                              # Align params
                              align_before_combine = align_before_combine, 
                              align_cut_outer = align_cut_outer,
                              align_outer_size = align_outer_size,
                              align_detection_sigma = align_detection_sigma,
                                    
                              # Subtraction params
                              subtract_before_photometry = subtract_before_photometry,
                              align_before_subtract = align_before_subtract,
                              cutout_target_image = cutout_target_image,
                              cutout_reference_image = cutout_reference_image,
                              cutout_size = cutout_size)
                result = pool.apply_async(self.photometry, kwds=kwargs)
                results.append(result)
            
            # Wait for all results to complete
            output = [result.get() for result in results]

        # Update target_imagelist
        self.target_imagelist = [item for sublist in output for item in sublist[0]]
        combined_dict = {}
        for dict_component in [sublist[1] for sublist in output]:
            for key, value in dict_component.items():
                if key not in combined_dict:
                    combined_dict[key] = []  # Initialize the key with an empty list if not present
                combined_dict[key].extend(value)  # Extend the list with new values
        self.failed_imagelist = combined_dict
        
        # Save
        if write_log:
            save_path = os.path.join(self.helper.photpath, 'photometry_log',datetime.now().strftime('%Y%m%d'))
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_photometry_failed_imagelist.txt", 'w') as file:
                json.dump(self.failed_imagelist, file, indent=4)
                print(f"Saved: {save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_photometry_failed_imagelist.txt")
        






























































































#%% KCT
if __name__ == '__main__':    
    filelist = sorted(glob.glob('/mnt/data1/supernova_rawdata/SN2021aefx/photometry/KCT_STX16803/g/*120.fits'))
    A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'KCT', ccd = 'STX16803'), reference_image = '/mnt/data1/reference_image/KCT_STX16803/Ref-KCT_STX16803-NGC1566-g-4440.com.fits')
    #filelist = sorted(glob.glob('/mnt/data1/supernova_rawdata/SN2021aefx/photometry/KCT_STX16803/r/Calib*120.fits'))
    #A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'KCT', ccd = 'STX16803'), reference_image = '/mnt/data1/reference_image/KCT_STX16803/Ref-KCT_STX16803-NGC1566-r-3360.com.fits')
    #filelist = sorted(glob.glob('/mnt/data1/supernova_rawdata/SN2021aefx/photometry/KCT_STX16803/i/Calib*120.fits'))
    #A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'KCT', ccd = 'STX16803'), reference_image = '/mnt/data1/reference_image/KCT_STX16803/Ref-KCT_STX16803-NGC1566-i-720.com.fits')
    sex_configfile = None
#%% LSGT
if __name__ == '__main__':    
    # filelist = sorted(glob.glob('/mnt/data1/supernova_rawdata/SN2021aefx/photometry/LSGT_SNUCAMII/g/Calib*180.fits'))
    # A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'LSGT', ccd = 'SNUCAMII'), reference_image = '/mnt/data1/reference_image/LSGT_STX16803/Calib-LSGT-NGC1566-20210916-180448-g-540.com.fits')
    #filelist = sorted(glob.glob('/mnt/data1/supernova_rawdata/SN2021aefx/photometry/LSGT_SNUCAMII/r/Calib*180.fits'))
    #A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'LSGT', ccd = 'SNUCAMII'), reference_image = '/mnt/data1/reference_image/LSGT_STX16803/Calib-LSGT-NGC1566-20210916-181452-r-540.com.fits')
    filelist = sorted(glob.glob('/mnt/data1/supernova_rawdata/SN2021aefx/photometry/LSGT_SNUCAMII/i/Calib*180.fits'))
    A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'LSGT', ccd = 'SNUCAMII'), reference_image = '/mnt/data1/reference_image/LSGT_STX16803/Calib-LSGT-NGC1566-20220401-100321-i-540.com.fits')
    sex_configfile = '/home/hhchoi1022/hhpy/Research/photometry/sextractor/LSGT_SNUCAM.config'
#%% RASA36
if __name__ == '__main__':    
    filelist = sorted(glob.glob('/mnt/data1/supernova_rawdata/SN2021aefx/photometry/RASA36/r/HIGH/Calib*60.fits'))
    filelist_1 = filelist[:592]
    filelist_2 = filelist[592:1003]
    filelist_3 = filelist[1003:1497]
    filelist_4 = filelist[1497:2016]
    filelist_5 = filelist[2016:]
    
#%% RASA36, continued...
if __name__ == '__main__':    
    A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'RASA36', ccd = 'KL4040', readoutmode = 'HIGH'), reference_image = '/mnt/data1/reference_image/RASA36_KL4040/Ref-RASA36-NGC1566-r-3180-HIGH.com.fits')
    sex_configfile = '/home/hhchoi1022/hhpy/Research/photometry/sextractor/RASA36_HIGH.config'
#%% RASA36, continued...
if __name__ == '__main__':    
    A = Photometry(filelist, telescope_info = Helper().get_telinfo(telescope = 'RASA36', ccd = 'KL4040', readoutmode = 'MERGE'), reference_image = '/mnt/data1/reference_image/RASA36_KL4040/Ref-RASA36-NGC1566-r-3180-MERGE.com.fits')
    sex_configfile = '/home/hhchoi1022/hhpy/Research/photometry/sextractor/RASA36_MERGE.config'
#%% Background subtraction
if __name__ == '__main__':
    A.faster_subtract_bkg(num_processes = 6,
                          filelist = None,
                          apply_2D_bkg = True,
                          mask_sources = True,
                          mask_source_size_in_pixel = 20,
                          bkg_estimator = 'median', # mean, median, sextractor, 
                          bkg_sigma = 3.0, 
                          bkg_box_size = 300, 
                          bkg_filter_size = 3, 
                          prefix_subbkg = 'subbkg_',
                          update_header = True,
                          visualize = True,
                          write_log = True,
                          print_output = True)
#%% Calculate ZP
if __name__ == '__main__':    
    #A.calculate(sex_configfile = sex_configfile, ref_catalog_name = 'APASS', ref_catalog_conversion = None, check_zp_by_color = True)
    A.faster_calculate(num_processes = 6,
                       filelist = None,
                       sex_configfile = sex_configfile,  # Absolute Path
                       detect_threshold = 3.0,
                       aperture_type = 'relative',  # relative or absolute
                       aperture_sizes = [1.5, 2.5, 3.5],  # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                       ref_catalog_name = 'APASS',
                       ref_catalog_conversion = None,
                       ref_maxmag = 15,
                       ref_minmag = 13,
                       visualize = False,
                       update_header = True,
                       check_zp_by_color = False,
                       write_log = True,
                       print_output = True
                       )
#%%
if __name__ == '__main__':    
    A.exclude_outlier(sigma = 5,
                      depth_cut = 17.0,
                      seeing_cut = 5.5,
                      depth_key = 'DEPTH5_APER_1_HH',
                      seeing_key = 'SEEING_HH',
                      visualize = True,
                      write_log = True,
                      move = False)
#%%
if __name__ == '__main__':    
    #A.combine(group_tolerance= 0.3)
    A.combine( # num_processes = 4,
                       group_key = 'jd',
                       group_tolerance = 0.2,
                       print_output = True,
                       
                       # Combine params
                       combine_method = 'median',
                       scale = 'multiply',
                       prefix_combine = 'com_',
                       zp_key ='ZP_APER_1_HH',
                       clip = 'extrema',
                       clip_sigma_low = 2,
                       clip_sigma_high = 5,
                       clip_minmax_min = 3,
                       clip_minmax_max = 3,
                       clip_extrema_nlow = 1,
                       clip_extrema_nhigh = 1,
                       
                       # Background Subtraction params
                       subbkg_before_combine = False,
                       apply_2D_bkg = False,
                       mask_sources = False,
                       mask_source_size_in_pixel = 10,
                       bkg_estimator = 'median', # mean, median, sextractor, 
                       bkg_sigma = 3.0, 
                       bkg_box_size = 300, 
                       bkg_filter_size = 3, 
                       prefix_subbkg = 'subbkg_',
                       update_header_subbkg = False,
                       visualize_subbkg = False,

                        # Calculate params
                       calculate_before_combine = False,
                       sex_configfile = sex_configfile, 
                       detect_threshold = 3.0,
                       aperture_type = 'relative',  # relative or absolute
                       aperture_sizes = [1.5, 2.5, 3.5],  # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                       ref_catalog_name = 'APASS',
                       ref_catalog_conversion = None,
                       ref_maxmag = 15.0,
                       ref_minmag = 12.0, 
                       visualize_calculate = False, 
                       update_header_calculate = False,
                       check_zp_by_color = False,
                        
                       # Align params
                       align_before_combine = True, 
                       align_cut_outer = True,
                       align_outer_size = 0.95,
                       align_detection_sigma = 5
                       )
#%%
if __name__ == '__main__':    
    A.photometry(ra = 64.9725, 
                 dec =  -54.948081, 
                 #num_processes = 4,
                 filelist = None,
                 sex_configfile = sex_configfile,
                 detect_threshold  = 3.0,
                 aperture_type = 'relative', # relative or absolute
                 aperture_sizes = [1.5, 2.5, 3.5], # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                 visualize_photometry = True,
                 write_log = True,
                 print_output = True,
                 
                 # Combine params
                 combine_before_photometry = False,
                 group_key = 'jd',
                 group_tolerance = 0.3,
                 zp_key ='ZP_APER_1_HH',
                 combine_method = 'median',
                 scale = 'multiply',
                 prefix_combine = 'com_',
                 clip = 'extrema',
                 clip_sigma_low = 2,
                 clip_sigma_high = 5,
                 clip_minmax_min = 3,
                 clip_minmax_max = 3,
                 clip_extrema_nlow = 1,
                 clip_extrema_nhigh = 1,
             
                 # Calculate params
                 calculate_before_photometry = True,
                 calculate_before_combine = False,
                 ref_catalog_name = 'APASS',
                 ref_catalog_conversion = None,
                 ref_maxmag= 15,
                 ref_minmag= 13,
                 visualize_calculate = False,
                 check_zp_by_color = False,
                 update_header_calculate = True,
 
                 # Background Subtraction params
                 subbkg_before_combine = False,
                 apply_2D_bkg = False,
                 mask_sources = False,
                 mask_source_size_in_pixel = 10,
                 bkg_estimator = 'median', # mean, median, sextractor, 
                 bkg_sigma = 3.0, 
                 bkg_box_size = 300, 
                 bkg_filter_size = 3, 
                 prefix_subbkg = 'subbkg_',
                 update_header_subbkg = True,
                 visualize_subbkg = False,
             
                 # Align params
                 align_before_combine = False, 
                 align_cut_outer = True,
                 align_outer_size = 0.95,
                 align_detection_sigma = 5,
                    
                 # Subtraction params
                 subtract_before_photometry = True,
                 align_before_subtract = True,
                 cutout_target_image = True,
                 cutout_reference_image = True,
                 cutout_size = 2500)

        # %%

# %%
