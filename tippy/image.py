
#%%
import glob
import inspect

from astropy.io import fits
import os
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, sigma_clip
import numpy as np
from astropy.table import Table
from astropy.time import Time
from Research.photometry import Catalog
from Research.helper import Helper
# %%
class Image:

    def __init__(self,
                 image: str,
                 telescope_info: dict = None,
                 reference_image : str = None
                 ):
        # Image information
        self.helper = Helper()
        self.original_image = image
        self.target_image = image
        self.reference_image = reference_image
        self._data = dict(data = None, image = '')
        self._header = dict(data = None, image = '')
        #self.header = fits.getheader(self.original_image)
        if telescope_info == None:
            telescope_info = self.helper.get_telinfo()
        self.telinfo = telescope_info

        self.catalog = None
        self.seeing = None
        self.zp = dict()
        self.zperr = dict()
        self.depth_3 = dict()
        self.depth_5 = dict()

    def __repr__(self):
        methods = [f'Image.{name}()\n' for name, method in inspect.getmembers(
            Image, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt

    @property
    def data(self):
        if self._data['image'] != self.target_image:
            self._data['data'] = fits.getdata(self.target_image)
            self._data['image'] = self.target_image            
        return self._data['data']
    
    @property
    def header(self):
        if self._header['image'] != self.target_image:
            self._header['data'] = fits.getheader(self.target_image)
            self._header['image'] = self.target_image            
        return self._header['data']

    def scamp(self, 
              sex_configfile : str = None,
              scamp_configfile : str = None,
              print_output : bool = True):
        self.helper.print(f'Start running SCAMP...', print_output)
        if sex_configfile == None:
            sex_configfile = self.helper.get_sexconfigpath(telescope = self.telinfo['obs'], ccd = self.telinfo['ccd'], readoutmode = self.telinfo['mode'], binning = self.telinfo['binning'], for_scamp = True)
        result = self.helper.run_scamp(filelist = self.target_image,
                                       sex_configfile = sex_configfile,
                                       scamp_configfile = scamp_configfile,
                                       update_files = True,
                                       print_output = False
                                       )
        self.helper.print(f'SCAMP is finished', print_output)
        
    def astrometry(self,
                   sex_configfile : str = None,
                   ra : float = None,
                   dec : float = None,
                   radius : float = None,
                   scalelow : float = 0.6, 
                   scalehigh : float = 0.8, 
                   overwrite : float =False, 
                   remove : bool = True,
                   print_output : bool = True
                   ):
        if sex_configfile == None:
            sex_configfile = self.helper.get_sexconfigpath(telescope = self.telinfo['obs'], ccd = self.telinfo['ccd'], readoutmode = self.telinfo['mode'], binning = self.telinfo['binning'])
        self.helper.print(f'Astrometry is started for {self.target_image}', print_output)
        if ra == None or dec == None:
            try:
                ra = float(self.header['RA'])
                dec = float(self.header['DEC'])
                radius = 2
            except:                
                try:
                    coord = SkyCoord(ra=self.header['RA'], dec=self.header['DEC'], unit=(u.hourangle, u.deg))
                    ra = coord.ra.value
                    dec =coord.dec.value
                    radius = 2
                except:
                    try:
                        self.helper.print('Searching RA, Dec from the header object name', print_output)
                        self.catalog = Catalog(target_name=self.header['OBJECT'])
                        ra = self.catalog.fieldinfo['ra']
                        dec = self.catalog.fieldinfo['dec']
                        radius = 2
                    except:
                        self.helper.print('RA, Dec is not available in the header. Astrometry without central coordinate', print_output)
                        pass
        
        self.target_image = self.helper.run_astrometry(image = self.target_image, 
                                                       sex_configfile = sex_configfile,
                                                       ra = ra,
                                                       dec = dec,
                                                       radius = radius,
                                                       scalelow = scalelow, 
                                                       scalehigh = scalehigh, 
                                                       overwrite = overwrite, 
                                                       remove = remove,
                                                       print_output = False)
        self.helper.print(f'Astrometry is finished for {self.target_image}', print_output)

    def calculate_zeropoint(self,
                            sex_configfile: str = None,  # Absolute Path
                            detect_threshold : float  = 3.0,
                            aperture_type : str = 'relative', # relative or absolute
                            aperture_sizes : list = [1.5, 2.5, 3.5], # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                            ref_catalog_name: str = 'APASS',
                            ref_catalog_conversion: str = None,
                            ref_maxmag=16,
                            ref_minmag=12,
                            visualize: bool = True,
                            update_header : bool = True,
                            check_zp_by_color : bool = False,
                            check_zp_correlation : bool = False,
                            correlation_key : str = 'X_IMAGE',
                            print_output : bool = True
                            ):
        '''
        sex_configfile: str = None  # Absolute Path
        detect_threshold : float  = 3.0
        aperture_type : str = 'relative' # relative or absolute
        aperture_sizes : list = [1.5, 2.5, 3.5] # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
        ref_catalog_name: str = 'APASS'
        ref_catalog_conversion: str = None
        ref_maxmag=16
        ref_minmag=12
        visualize: bool = True
        update_header : bool = True
        check_zp_by_color : bool = True
        print_output = True
        '''

        self.helper.print(f'Calculating zeropoint for {self.target_image}', print_output)
        if sex_configfile == None:
            sex_configfile = self.helper.get_sexconfigpath(telescope = self.telinfo['obs'], ccd = self.telinfo['ccd'], readoutmode = self.telinfo['mode'], binning = self.telinfo['binning'])
        sex_params = dict()
        sex_params['CATALOG_NAME'] = f"{self.helper.sexpath}/result/{os.path.basename(self.target_image).split('.')[0]}.cat"
        os.makedirs(os.path.dirname(sex_params['CATALOG_NAME']), exist_ok=True)
        
        # Run source-extractor with the default configuration files
        obs_catalog_temp1 = self.helper.run_sextractor(image=self.target_image, sex_configfile=sex_configfile, sex_params=sex_params, return_result=True, print_output = False)
        source_idx = obs_catalog_temp1['FWHM_IMAGE'] * self.telinfo['pixelscale'] > 1.8
        obs_catalog_temp1_source = obs_catalog_temp1[source_idx]
        
        # Applying rough point source criteria
        _, seeing_temp, _ = sigma_clipped_stats(data=obs_catalog_temp1_source['FWHM_IMAGE'], sigma_lower=3, sigma_upper=2, maxiters=5, cenfunc='median', stdfunc='std')
        seeing_temp_pixel = np.round(seeing_temp, 2)
        seeing_temp_arcsec = np.round(seeing_temp_pixel * self.telinfo['pixelscale'], 2)
        
        # Run source-extractor with the updated FWHM information
        sex_params['SEEING_FWHM'] = str(seeing_temp_arcsec)
        sex_params['DETECT_MINAREA'] = str(int(np.pi * (seeing_temp_pixel/2)**2))
        obs_catalog_temp2 = self.helper.run_sextractor(image=self.target_image, sex_configfile=sex_configfile, sex_params=sex_params, return_result=True, print_output = False)        
        
        # Applying detailed point source criteria
        star_idx1 = (obs_catalog_temp2['FLAGS'] < 4) & (obs_catalog_temp2['CLASS_STAR'] > 0.8) & (obs_catalog_temp2['FWHM_WORLD']*3600 < seeing_temp_arcsec*1.5)
        obs_catalog_temp2_star1 = obs_catalog_temp2[star_idx1]
        star_idx2 = sigma_clip(data=obs_catalog_temp2_star1['FWHM_IMAGE'], sigma_lower=3, sigma_upper=2, maxiters=5, cenfunc='median', stdfunc='std').mask
        obs_catalog_temp2_star2 = obs_catalog_temp2_star1[~star_idx2]
        
        # Seeing measurement
        seeing_median = np.median(obs_catalog_temp2_star2['FWHM_WORLD']*3600)
        self.seeing = seeing_median
        
        # Run source-extractor with the updated aperture information 
        sex_params['SEEING_FWHM'] = str(seeing_temp_arcsec)
        sex_params['DETECT_THRESH'] = detect_threshold
        sex_params['DETECT_MINAREA'] = str(int(np.pi * (self.seeing/2)**2))
        sex_params['ANALYSIS_THRESH'] = detect_threshold
        if aperture_type.upper() == 'RELATIVE':
            aperture_size_list = [str(aperture_factor * self.seeing/ self.telinfo['pixelscale']) for aperture_factor in aperture_sizes]
        else:
            aperture_size_list = [str(aperture_factor / self.telinfo['pixelscale']) for aperture_factor in aperture_sizes]
        sex_params['PHOT_APERTURES'] = ','.join(aperture_size_list)
        
        obs_catalog = self.helper.run_sextractor(image=self.target_image, sex_configfile=sex_configfile, sex_params=sex_params, return_result=True, print_output = False)        
        # Applying detailed point source criteria
        star_idx = (obs_catalog['FLAGS'] < 4) & (obs_catalog['CLASS_STAR'] > 0.9) #& (obs_catalog['FWHM_WORLD']*3600 < self.seeing*1.2)
        obs_catalog_star = obs_catalog[star_idx]
        non_star_idx_selected = sigma_clip(data=obs_catalog_star['FWHM_IMAGE'], sigma_lower=3, sigma_upper=2, maxiters=10, cenfunc='median', stdfunc='std').mask
        obs_catalog_star = obs_catalog_star[~non_star_idx_selected]
        seeing_median = np.median(obs_catalog_star['FWHM_WORLD']*3600)
        self.seeing = seeing_median

        # Calculate ZP
        ## Load reference catalog
        if self.catalog == None:
            self.catalog = Catalog(target_name=self.header['OBJECT'])
        ref_catalog = self.catalog.dict[ref_catalog_name].data
        if ref_catalog_name == 'GAIA':
            ref_catalog = ref_catalog[ref_catalog['c_star'] < 0.05]
        if ref_catalog_conversion:
            try:
                ref_catalog = self.catalog.dict[ref_catalog_name].conversion[ref_catalog_conversion]
            except KeyError as e:
                raise KeyError(f"Conversion catalog {ref_catalog_conversion} is not available: Available conversion of {ref_catalog_name} = [{list(self.catalog.dict[ref_catalog_name].conversion.keys())}]")
        
        ## Cross-match obs_catalog & sky_catalog
        idx_obs, idx_ref, dist_second = self.helper.cross_match(obj_catalog=SkyCoord(obs_catalog_star['ALPHA_J2000'], obs_catalog_star['DELTA_J2000'], unit='deg'), sky_catalog=SkyCoord(ra=ref_catalog['ra'], dec=ref_catalog['dec'], unit='deg'), max_distance_second= self.seeing * 1)
        
        obs_matched = obs_catalog_star[idx_obs]
        ref_matched = ref_catalog[idx_ref]

        ## Calculate
        mag_ref_key = f"{self.header['FILTER']}_mag"
        magerr_ref_key = f"e_{self.header['FILTER']}_mag"
        for i, aperture_size in enumerate(aperture_size_list):
            mag_obs_key = 'MAG_APER' if i == 0 else f'MAG_APER_{i}'
            magerr_obs_key = 'MAGERR_APER' if i == 0 else f'MAGERR_APER_{i}'
            aper_key = 'APER' if i == 0 else f'APER_{i}'
            # Temporary ZP calculation
            mag_diff_temp = ref_matched[mag_ref_key] - obs_matched[mag_obs_key]
            zp_temp = np.median(mag_diff_temp)
            # ZP star temporary cut
            zp_star_idx1 = (ref_minmag < obs_matched[mag_obs_key] + zp_temp) & (obs_matched[mag_obs_key] + zp_temp < ref_maxmag)
            mag_diff_temp2 = ref_matched[zp_star_idx1][mag_ref_key] - obs_matched[zp_star_idx1][mag_obs_key]
            # Sigma clipping
            zp_star_idx2 = sigma_clip(data=mag_diff_temp2, sigma_lower=5, sigma_upper=5, maxiters=3, cenfunc='median', stdfunc='std').mask
            # ZP/Depth Calculation with selected sources 
            obs_selected = obs_matched[zp_star_idx1][~zp_star_idx2]
            ref_selected = ref_matched[zp_star_idx1][~zp_star_idx2]
            zp = ref_selected[mag_ref_key] - obs_selected[mag_obs_key]
            zp_median = np.median(zp)
            #zperr_ref = np.sqrt(np.abs(ref_selected[magerr_ref_key])**2 + np.abs(obs_selected[magerr_obs_key])**2)
            #zperr = np.sqrt(np.sum(zperr_ref**2)) / len(zperr_ref)
            zperr = np.std(zp)
            depth_5sig = round(-2.5*np.log10(5*(obs_catalog_star['THRESHOLD'][0]/detect_threshold)*np.sqrt(np.pi*((float(aperture_size)/2)**2))) + zp_median,3)
            depth_3sig = round(-2.5*np.log10(3*(obs_catalog_star['THRESHOLD'][0]/detect_threshold)*np.sqrt(np.pi*((float(aperture_size)/2)**2))) + zp_median,3)
            self.zp[aper_key] = np.round(zp_median,5)
            self.zperr[aper_key] = np.round(zperr,5)
            self.depth_3[aper_key] = np.round(depth_3sig,5)
            self.depth_5[aper_key] = np.round(depth_5sig,5)
            
            if check_zp_correlation:
                correlation_table = Table()
                correlation_table['ZP'] = zp
                correlation_table['MAG_INST'] = obs_selected[mag_obs_key]
                correlation_table['MAG_REF'] = ref_selected[mag_ref_key]
                correlation_table['FWHM'] = obs_selected['FWHM_IMAGE']
                correlation_table['X_IMAGE'] = obs_selected['X_IMAGE']
                correlation_table['Y_IMAGE'] = obs_selected['Y_IMAGE']
                correlation_table['g-r'] = ref_selected['g_mag'] - ref_selected['r_mag']
                df = correlation_table.to_pandas()
                correlation_matrix = df.corr(method = 'pearson')
                
                import seaborn as sns
                import matplotlib.pyplot as plt

                plt.figure(figsize  =(10,8))
                plt.title(f'APERTURE_{i} = {aperture_size}')
                sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.1f', linewidth = 0.5)
                plt.show()
                
                plt.scatter(correlation_table[correlation_key], zp, c='k', alpha = 0.7, marker = 'o')
            
            
            
            if check_zp_by_color:
                if i == 1:
                    import matplotlib.pyplot as plt
                    plt.title(f'Catalog = {ref_catalog_name} / Conversion = {ref_catalog_conversion}')
                    plt.scatter(ref_matched['g_mag'] - ref_matched['r_mag'], zp_matched, c='k', alpha = 0.2, marker = 'o')
                    plt.scatter(ref_selected['g_mag'] - ref_selected['r_mag'], zp, c='r', alpha = 0.2, marker = 'o')
                    plt.axhline(zp_median, c='r', linestyle='--', label=f'ZP = {np.round(zp_median, 1)}')
                    plt.ylim(zp_median - 0.2, zp_median + 0.2)
                    plt.xlim(-0.5,2)
                    plt.xlabel(rf'g-r')
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(np.array(ref_selected['g_mag'] - ref_selected['r_mag']).reshape(-1, 1), zp)
                    slope = model.coef_[0]
                    intercept = model.intercept_
                    # Plot the linear fit
                    x_fit = np.linspace(-0.5, 2, 100)
                    y_fit = model.predict(x_fit.reshape(-1, 1))
                    plt.plot(x_fit, y_fit, color='blue', linestyle='-', linewidth=2, label=f'Fit: y = {slope:.2f}(g-r) + {intercept:.2f}')
                    plt.legend()
                    plt.show()
        
        self.helper.print(f'ZP = {self.zp}, SEEING = {self.seeing}, DEPTH_3 = {self.depth_3}, DEPTH_5 = {self.depth_5}', print_output)

        if update_header:
            hdul = fits.open(self.target_image, mode='update')
            header = hdul[0].header

            for i, aperture_size in enumerate(aperture_size_list):
                zp_key = 'ZP_APER' if i == 0 else f'ZP_APER_{i}'
                zperr_key = 'ZPERR_APER' if i == 0 else f'ZPERR_APER_{i}'
                depth3_key = 'DEPTH3_APER' if i == 0 else f'DEPTH3_APER_{i}'
                depth5_key = 'DEPTH5_APER' if i == 0 else f'DEPTH5_APER_{i}'
                aper_key = 'APER' if i == 0 else f'APER_{i}'
                aper_size_key = 'APER_SIZE' if i == 0 else f'APER_SIZE_{i}'
                seeing_key = 'SEEING_HH'
                
                # Update or add a new value to the header
                header[f'{seeing_key}'] = self.seeing
                header[f'{zp_key}_HH'] = self.zp[aper_key]
                header[f'{zperr_key}_HH'] = self.zperr[aper_key]
                header[f'{depth3_key}_HH'] = self.depth_3[aper_key]
                header[f'{depth5_key}_HH'] = self.depth_5[aper_key]
                header[f'{aper_size_key}_HH'] = aperture_size

            # Save the changes and close the FITS file
            hdul.flush()
            hdul.close()    
            
        if visualize:
            import matplotlib.pyplot as plt

            # Create a figure with two subplots, arranged vertically
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

            # First subplot: SEEING
            ax1.set_title('SEEING')
            ax1.scatter(obs_catalog['MAG_APER_1'] + self.zp['APER_1'], obs_catalog['FWHM_WORLD'] * 3600, c='k', alpha = 0.5)
            ax1.scatter(obs_selected['MAG_APER_1'] + self.zp['APER_1'], obs_selected['FWHM_WORLD'] * 3600, c='r', alpha = 0.5)
            ax1.axhline(seeing_median, c='r', linestyle='--', label=f'Seeing = {np.round(seeing_median, 1)}')
            ax1.set_xlim(9, 20)
            ax1.set_ylim(seeing_median - 2, seeing_median + 3)
            ax1.set_xlabel('MAG_APER_1')
            ax1.set_ylabel('FWHM (arcsec)')
            ax1.legend()

            # Second subplot: ZP
            ax2.set_title(f'ZP [Aperture = {np.round(self.telinfo["pixelscale"] *float(aperture_size_list[1]),2)}"]')
            ax2.scatter(obs_matched['MAG_APER_1'] + self.zp['APER_1'], ref_matched[mag_ref_key] - obs_matched['MAG_APER_1'], c='k', alpha = 0.5)
            ax2.scatter(obs_selected['MAG_APER_1'] + self.zp['APER_1'], ref_selected[mag_ref_key] - obs_selected['MAG_APER_1'], c='r', alpha = 0.5)
            ax2.axhline(self.zp['APER_1'], c='r', linestyle='--', label=f'ZP = {np.round(self.zp["APER_1"], 1)}')
            ax2.set_xlim(9, 20)
            ax2.set_ylim(self.zp["APER_1"] - 2, self.zp["APER_1"] + 3)
            ax2.set_xlabel('MAG_APER_1')
            ax2.set_ylabel('ZP')
            ax2.legend()

            # Adjust the layout to make space for titles and labels
            fig.tight_layout()
            outputfile = self.target_image.split('fit')[0]+'_calc.png'
            fig.savefig(outputfile)

            # Show the combined figure
            plt.show()
        self.calculated = True  

    def align(self, cut_outer : bool = False, outer_size = 0.95, detection_sigma = 5, print_output : bool = True):
        if not self.reference_image:
            raise ValueError('Reference image is requred for image alignment')
        self.helper.print(f'Aligning {self.target_image} with {self.reference_image}', print_output)
        if cut_outer:
            self.target_image = self.helper.cutout_img(target_img = self.target_image, size = outer_size, prefix = 'cutouter_', print_output = False)
            self.reference_image = self.helper.cutout_img(target_img = self.reference_image, size = outer_size, prefix = 'cutouter_', print_output = False)
        self.target_image = self.helper.align_img(target_img=self.target_image, reference_img=self.reference_image, detection_sigma= detection_sigma, print_output = False)
        self.helper.print(f'Aligned image is saved as {self.target_image}', print_output)
        
    def cutout(self, cutout_size = 2000, print_output : bool = True):
        self.helper.print(f'Cutting out the image {self.target_image} with size of {cutout_size}', print_output)
        self.target_image = self.helper.cutout_img(target_img = self.target_image, size = cutout_size, prefix = 'cutout_')
        self.helper.print(f'Cutout image is saved as {self.target_image}', print_output)
        
    def subtract(self,
                 align : bool = True,
                 reference_mask : str = None,
                 cutout_target_image : bool = True,
                 cutout_reference_image : bool = True,
                 cutout_size : int = 1500,
                 print_output : bool = True
                 ):
        if not self.reference_image:
            raise ValueError('Reference image is requred for image subtraction')
        self.helper.print(f'Subtracting {self.target_image} with {self.reference_image}', print_output)
        if align:
            self.align(cut_outer = False, print_output = False)
        if cutout_target_image:
            self.cutout(cutout_size = cutout_size, print_output = False)
        if cutout_reference_image:
            self.reference_image = self.helper.cutout_img(target_img = self.reference_image, size = cutout_size, print_output = False)
            reference_mask = self.helper.cutout_img(target_img = reference_mask, size = cutout_size, print_output = False)
        self.target_image = self.helper.subtract_img(target_img=self.target_image, reference_img=self.reference_image, reference_mask = reference_mask, print_output = False)
        self.helper.print(f'Subtracted image is saved as {self.target_image}', print_output)
    
    def subtract_bkg(self, 
                     apply_2D_bkg: bool = True,
                     mask_sources: bool = False,
                     mask_source_size_in_pixel : int = 10,
                     bkg_estimator: str = 'median', # mean, median, sextractor, 
                     bkg_sigma: float = 3.0, 
                     bkg_box_size: int = 300, 
                     bkg_filter_size: int = 3, 
                     prefix : str = 'subbkg_',
                     update_header: bool = True,
                     visualize: bool = False,
                     print_output : bool = True):
        self.helper.print(f'Subtracting background from {self.target_image}', print_output)
        self.target_image = self.helper.subtract_background(target_img=self.target_image, apply_2D_bkg = apply_2D_bkg, mask_sources = mask_sources, mask_source_size_in_pixel = mask_source_size_in_pixel, bkg_estimator = bkg_estimator, bkg_sigma = bkg_sigma, bkg_box_size = bkg_box_size, bkg_filter_size = bkg_filter_size, prefix = prefix, update_header = update_header, visualize = visualize, print_output = False)
        self.helper.print(f'Subtracted background image is saved as {self.target_image}', print_output)
    
    def photometry(self,
                   ra : float,
                   dec : float,
                   sex_configfile: str = None,
                   detect_threshold : float  = 3.0,
                   aperture_type : str = 'relative', # relative or absolute
                   aperture_sizes : list = [1.5, 2.5, 3.5], # relative (1.5*seeing, 2.5*seeing, 3.5*seeing) or absolute (3", 5", 7")
                   #cutout_target_image : bool = True,
                   #cutout_reference_image : bool = False,
                   #cutout_size : int = 1500,
                   #align : bool = True,
                   #subtract : bool = True,
                   #subtract_bkg : bool = True,
                   visualize : bool = True,
                   print_output : bool = True
                   ):
        '''
        ra =64.9723704
        dec = -54.9481347
        sex_configfile: str = None
        detect_threshold : float  = 3.0
        aperture_type : str = 'relative'
        aperture_sizes : list = [1.5, 2.5, 3.5]
        cutout_target_image : bool = True
        cutout_reference_image : bool = True
        cutout_size : int = 1500
        align : bool = True,
        subtract : bool = True
        visualize : bool = True
        '''
        
        self.helper.print(f'Performing photometry for {self.target_image}', print_output)
        try:
            for i, aperture_size in enumerate(aperture_sizes):
                zp_key = 'ZP_APER' if i == 0 else f'ZP_APER_{i}'
                zperr_key = 'ZPERR_APER' if i == 0 else f'ZPERR_APER_{i}'
                depth3_key = 'DEPTH3_APER' if i == 0 else f'DEPTH3_APER_{i}'
                depth5_key = 'DEPTH5_APER' if i == 0 else f'DEPTH5_APER_{i}'
                aper_key = 'APER' if i == 0 else f'APER_{i}'
                aper_size_key = 'APER_SIZE' if i == 0 else f'APER_SIZE_{i}'
                seeing_key = 'SEEING_HH'
                
                # Update or add a new value to the header
                self.seeing = self.header[seeing_key]
                self.zp[aper_key] = self.header[f'{zp_key}_HH']
                self.zperr[aper_key] = self.header[f'{zperr_key}_HH']
                self.depth_3[aper_key] = self.header[f'{depth3_key}_HH']
                self.depth_5[aper_key] = self.header[f'{depth5_key}_HH']               
        except:
            raise ValueError('Zeropoint information is not available in the header. Run calculate_zeropoint()')             
            
        #if subtract:
        #    self.helper.print(f'Subtracting background from {self.target_image}', print_output)
        #    self.subtract(align = align, cutout_target_image = cutout_target_image, cutout_reference_image = cutout_reference_image, cutout_size = cutout_size, print_output = False)
        
        #if subtract_bkg:
        #    self.helper.print(f'Subtracting background from {self.target_image}', print_output)
        #    self.subtract_bkg(target_img = self.target_image, print_output = False)
        
        # source-extractor configuration
        if sex_configfile == None:
            sex_configfile = self.helper.get_sexconfigpath(telescope = self.telinfo['obs'], ccd = self.telinfo['ccd'], readoutmode = self.telinfo['mode'], binning = self.telinfo['binning'])
        sex_params = dict()
        sex_params['CATALOG_NAME'] = f"{self.helper.sexpath}/result/{os.path.basename(self.target_image).split('.')[0]}.phot.cat"
        sex_params['SEEING_FWHM'] = str(self.seeing)
        sex_params['DETECT_THRESH'] = detect_threshold
        sex_params['ANALYSIS_THRESH'] = detect_threshold
        if aperture_type.upper() == 'RELATIVE':
            aperture_size_list = [str(aperture_factor * self.seeing/ self.telinfo['pixelscale']) for aperture_factor in aperture_sizes]
        else:
            aperture_size_list = [str(aperture_factor / self.telinfo['pixelscale']) for aperture_factor in aperture_sizes]
        sex_params['PHOT_APERTURES'] = ','.join(aperture_size_list)
        
        obs_catalog = self.helper.run_sextractor(image=self.target_image, sex_configfile=sex_configfile, sex_params=sex_params, return_result=True)

        idx_obs, idx_ref, dist_second = self.helper.cross_match(obj_catalog=SkyCoord(ra = obs_catalog['ALPHA_J2000'], dec = obs_catalog['DELTA_J2000'], unit='deg', frame = 'icrs'), sky_catalog=SkyCoord(ra=[ra], dec=[dec], unit='deg', frame = 'icrs'), max_distance_second=self.seeing * 3)
        
        phot_info = dict(zip(obs_catalog.colnames, len(obs_catalog.colnames)*[[None]]))
        detected = False
            
        ## Calculate
        target = obs_catalog[idx_obs]
        for i, aperture_size in enumerate(aperture_size_list):
            mag_obs_key = 'MAG_APER' if i == 0 else f'MAG_APER_{i}'
            magerr_obs_key = 'MAGERR_APER' if i == 0 else f'MAGERR_APER_{i}'
            mag_sky_key = 'MAG_APER_SKY' if i == 0 else f'MAG_APER_SKY_{i}'
            magerr_sky_key = 'MAGERR_APER_SKY' if i == 0 else f'MAGERR_APER_SKY_{i}'
            zp_key = 'ZP_APER' if i == 0 else f'ZP_APER_{i}'
            zperr_key = 'ZPERR_APER' if i == 0 else f'ZPERR_APER_{i}'
            depth3_key = 'DEPTH3_APER' if i == 0 else f'DEPTH3_APER_{i}'
            depth5_key = 'DEPTH5_APER' if i == 0 else f'DEPTH5_APER_{i}'
            aper_key = 'APER' if i == 0 else f'APER_{i}'
            aper_size_key = 'APER_SIZE' if i == 0 else f'APER_SIZE_{i}'
            phot_info[zp_key] = [self.zp[aper_key]]
            phot_info[zperr_key] = [self.zperr[aper_key]]
            phot_info[depth3_key] = [self.depth_3[aper_key]]
            phot_info[depth5_key] = [self.depth_5[aper_key]]
            phot_info[aper_size_key] = [aperture_size]
            phot_info['DATE-OBS'] = [Time(self.header['DATE-OBS']).datetime]
            phot_info['JD'] = [self.header['JD']]
            if len(target) > 0:
                phot_info[mag_sky_key] = [target[mag_obs_key][0] + self.zp[aper_key]]
                phot_info[magerr_sky_key] = [np.sqrt(target[magerr_obs_key][0]**2 + self.zperr[aper_key]**2)]
                detected = True
        if detected:
            self.helper.print(f'(Detected) Photometry is done for {self.target_image}', print_output)
        else:
            self.helper.print(f'(Non-detected) Photometry is done for {self.target_image}', print_output)
        phot_tbl = Table(phot_info)
        outputfile = self.target_image.split('fit')[0]+'phot'
        phot_tbl.write(outputfile, format = 'ascii', overwrite = True)
        
        if visualize:
            import matplotlib.pyplot as plt
            from astropy.wcs import WCS
            from matplotlib.patches import Circle
            
            # Load data
            hdu = fits.open(name = self.target_image)[0]
            data = hdu.data
            wcs = WCS(hdu.header)
            
            #central_pixel = float(target['X_IMAGE']), float(target['Y_IMAGE'])
            central_pixel = wcs.world_to_pixel(SkyCoord(ra, dec, unit = 'deg', frame = 'icrs'))
            
            # Define the region to visualize
            x_min = int(central_pixel[0] - 25 / self.telinfo['pixelscale'])
            x_max = int(central_pixel[0] + 25 / self.telinfo['pixelscale'])
            y_min = int(central_pixel[1] - 25 / self.telinfo['pixelscale'])
            y_max = int(central_pixel[1] + 25 / self.telinfo['pixelscale'])
            region_data = data[y_min:y_max, x_min:x_max]
            region_wcs = wcs[y_min:y_max, x_min:x_max]
            
            # Plot the extracted region
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection=region_wcs)
            ax.imshow(region_data, origin='lower', cmap='gray', aspect='auto')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.set_title(f'Region around ({ra}, {dec})')
            ax.grid(color='white', ls='solid')

            # Draw the aperture
            aperture_radius_pixels   =  0.5 * self.seeing/ self.telinfo['pixelscale']
            aperture15_radius_pixels =  1.5 * self.seeing/ self.telinfo['pixelscale']
            aperture25_radius_pixels =  2.5 * self.seeing/ self.telinfo['pixelscale']
            aperture35_radius_pixels =  3.5 * self.seeing/ self.telinfo['pixelscale']
            aperture15_circle = Circle((central_pixel[0] - x_min, central_pixel[1] - y_min), aperture15_radius_pixels, transform=ax.get_transform('pixel'), color='r', fill=False, lw=2, label  =r'1.5$\times$Seeing')
            aperture25_circle = Circle((central_pixel[0] - x_min, central_pixel[1] - y_min), aperture25_radius_pixels, transform=ax.get_transform('pixel'), color='b', fill=False, lw=2, label  =r'2.5$\times$Seeing')
            aperture35_circle = Circle((central_pixel[0] - x_min, central_pixel[1] - y_min), aperture35_radius_pixels, transform=ax.get_transform('pixel'), color='g', fill=False, lw=2, label  =r'3.5$\times$Seeing')
            if detected:
                ax.scatter(central_pixel[0] - x_min, central_pixel[1] - y_min, s = 80*aperture_radius_pixels, marker = '*', c = 'r', label = 'Target(Detection)')
            else:
                ax.scatter(central_pixel[0] - x_min, central_pixel[1] - y_min, s = 80*aperture_radius_pixels, marker = '*', c = 'b', label = 'Target(Non-detection)')

            ax.add_patch(aperture15_circle)
            ax.add_patch(aperture25_circle)
            ax.add_patch(aperture35_circle)
            ax.set_xlim()
            plt.legend(loc = 3)
            outputfile = self.target_image.split('fit')[0]+'_phot.png'
            plt.savefig(outputfile)            
        
    def visualize(self):
        self.helper.visualize_image(self.target_image)
#%% KCT
if __name__ == '__main__':
    imagelist = glob.glob('/data1/supernova_rawdata/SN2021aefx/photometry/KCT_STX16803/g/Calib*.fits')
    sci_image = imagelist[300]#'/data1/supernova_rawdata/SN2021aefx/photometry/KCT_STX16803/g/Calib-KCT_STX16803-NGC1566-20211113-054040-g-120.fits'
    ref_image = '/data1/supernova_rawdata/SN2021aefx/photometry/KCT_STX16803/Ref-KCT_STX16803-NGC1566-g-4440.com.fits'
    phot_helper = Helper()
    telinfo = phot_helper.get_telinfo(telescope='KCT', ccd='STX16803')
    im = Image(sci_image, telescope_info=telinfo, reference_image = ref_image)
    im.calculate_zeropoint(check_zp_correlation= True, correlation_key = 'FWHM')
# %%
