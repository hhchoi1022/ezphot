
#%%
from ezphot.imageobjects import Mask, Errormap, Background, CalibrationImage, MasterImage
from typing import Union, List
from astropy.time import Time
from typing import Optional
import numpy as np
#%%

class ImageMethod:    
    def correct_bias(self,
                     bias_image: Union[CalibrationImage, MasterImage] = None,
                     save: bool = False,
                     verbose: bool = True
                     ):
        
        """
        Correct bias in the image.
        
        Parameters
        ----------
        bias_image : CalibrationImage or MasterImage
            The bias image to correct bias.
        save : bool, optional
            Whether to save the corrected image.

        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """
        from ezphot.methods import Preprocess
        if bias_image is None:
            bias_image = self.get_masterframe(imagetyp = 'BIAS', max_days = 100)
        preprocess = Preprocess()
        calib_image = preprocess.correct_bias(
            target_img = self,
            bias_image = bias_image,
            save = save,
            verbose = verbose
        )
        return calib_image
    
    def correct_dark(self,
                     dark_image: Union[CalibrationImage, MasterImage] = None, 
                     save : bool = False,
                     verbose: bool = True
                     ):
        """
        Correct dark in the image.
        
        Parameters
        ----------
        dark_image : CalibrationImage or MasterImage
            The dark image to correct dark.
        save : bool, optional
            Whether to save the corrected image.
        
        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """
        from ezphot.methods import Preprocess
        if dark_image is None:
            dark_image = self.get_masterframe(imagetyp = 'DARK', max_days = 100)
        preprocess = Preprocess()
        calib_image = preprocess.correct_dark(
            target_img = self,
            dark_image = dark_image,
            save = save,
            verbose = verbose
        )
        return calib_image
    
    def correct_flat(self,
                     flat_image: Union[CalibrationImage, MasterImage] = None,
                     save: bool = False,
                     verbose: bool = True
                     ):
        """
        Correct flat in the image.
        
        Parameters
        ----------
        flat_image : CalibrationImage or MasterImage    
            The flat image to correct flat.
        save : bool, optional
            Whether to save the corrected image.

        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """
        from ezphot.methods import Preprocess
        if flat_image is None:
            flat_image = self.get_masterframe(imagetyp = 'FLAT', max_days = 100)
        preprocess = Preprocess()
        calib_image = preprocess.correct_flat(
            target_img = self,
            flat_image = flat_image,
            save = save,
            verbose = verbose,
        )
        return calib_image
    
    def correct_bdf(self,
                    bias_image: Union[CalibrationImage, MasterImage] = None,
                    dark_image: Union[CalibrationImage, MasterImage] = None,
                    flat_image: Union[CalibrationImage, MasterImage] = None,
                    save: bool = False,
                    verbose: bool = True
                    ):
        """
        Correct bias, dark, and flat.
        
        Parameters
        ----------
        bias_image : CalibrationImage or MasterImage
            The bias image to correct bias, dark, and flat.
        dark_image : CalibrationImage or MasterImage
            The dark image to correct bias, dark, and flat.
        flat_image : CalibrationImage or MasterImage
            The flat image to correct bias, dark, and flat.
        save : bool, optional
            Whether to save the corrected image.

        Returns
        -------
        calib_image : ScienceImage
            The corrected image.
        """
        from ezphot.methods import Preprocess
        if bias_image is None:
            bias_image = self.get_masterframe(imagetyp = 'BIAS', max_days = 100)
        if dark_image is None:
            dark_image = self.get_masterframe(imagetyp = 'DARK', max_days = 100)
        if flat_image is None:
            flat_image = self.get_masterframe(imagetyp = 'FLAT', max_days = 100)
        preprocess = Preprocess()
        calib_image = preprocess.correct_bdf(
            target_img = self,
            bias_image = bias_image,
            dark_image = dark_image,
            flat_image = flat_image,
            save = save,
            verbose = verbose
        )
        return calib_image
    
    def platesolve(self,
                   overwrite: bool = True,
                   verbose: bool = True,
                   scamp_sexparams: dict = None,
                   scamp_params: dict = None
                   ):
        """
        Solve astrometry using Astrometry.net.
        
        Parameters
        ----------
        target_img : Union[ScienceImage, ReferenceImage]
            The target image to solve astrometry for.
        overwrite : bool, optional
            Whether to overwrite the existing astrometry solution.
        verbose : bool, optional
            Whether to print verbose output.
            
        Returns
        -------
        output_img : ScienceImage
            The output image with astrometry solution.
        """
        from ezphot.methods import Platesolve
        platesolve = Platesolve()
        output_img = platesolve.solve_astrometry(
            target_img = self,
            overwrite = overwrite,
            verbose = verbose
            )
        
        output_img = platesolve.solve_scamp(
            target_img = self,
            scamp_sexparams = scamp_sexparams,
            scamp_params = scamp_params,
            overwrite = overwrite,
            verbose = verbose
        )
        
        return output_img
        
    def calculate_invalidmask(self,
                              threshold_invalid_connection: int = 100000,
                              save: bool = False,
                              verbose: bool = True,
                              visualize: bool = True,
                              save_fig: bool = False
                              ):
        """ 
        Calculate the invalid mask for this ScienceImage.
        The invalid mask is a mask of pixels that are invalid (zero or nan value).
        If save is True, the invalid mask is saved. Then, you can load the invalid mask with `self.invalidmask`.
        
        Parameters
        ----------
        threshold_invalid_connection : int
            The threshold for invalid connection.
        save : bool
            If True, save the invalid mask.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the invalid mask.
        save_fig : bool
            If True, save the figure of the invalid mask.
            
        Returns
        -------
        target_ivpmask : Mask
            The invalid mask.
        """
        from ezphot.methods import MaskGenerator
        maskgenerator = MaskGenerator()
        target_ivpmask = maskgenerator.mask_invalidpixel(
            target_img = self,
            threshold_invalid_connection= threshold_invalid_connection,
            # Others
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig
            )
        return target_ivpmask
    
    def calculate_circularmask(self,
                               target_srcmask: Mask = None,
                               x_position: float = None,
                               y_position: float = None,
                               radius_arcsec: float = None,
                               unit = 'coord',
                               save: bool = False,
                               verbose: bool = True,
                               visualize: bool = True,
                               save_fig: bool = False
                               ):
        """
        Calculate the circular mask for this ScienceImage.
        The circular mask is a mask of pixels that are within a circular region.
        If save is True, the circular mask is saved. Then, you can load the circular mask with `self.circularmask`.
        
        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        x_position : float
            The x position of the center of the circular mask.
        y_position : float
            The y position of the center of the circular mask.
        radius_arcsec : float
            The radius of the circular mask in arcseconds.
        unit : str
            The unit of the x and y position. 'coord' for coordinate, 'pixel' for pixel.
        save : bool 
            If True, save the circular mask.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the circular mask.
        save_fig : bool
            If True, save the figure of the circular mask.
        
        Returns 
        -------
        target_circularmask : Mask
            The circular mask.
        """
        from ezphot.methods import MaskGenerator
        maskgenerator = MaskGenerator()
        target_sourcemask = maskgenerator.mask_circle(
            target_img = self,
            target_mask = target_srcmask,
            mask_type = 'source',
            x_position = x_position,
            y_position = y_position,
            radius_arcsec = radius_arcsec,
            unit = unit,
            
            # Others
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig
            )
        return target_sourcemask
        
    def calculate_sourcemask(self,
                             target_srcmask: Mask = None,
                             sigma: float = 5.0,
                             mask_radius_factor: float = 3,
                             saturation_level: float = 50000,
                             save: bool = False,
                             verbose: bool = True,
                             visualize: bool = True,
                             save_fig: bool = False
                             ):
        """ 
        Calculate the source mask for this ScienceImage.
        The source mask is a mask of pixels that are sources. Detection is made with global background and background RMS map.
        If save is True, the source mask is saved. Then, you can load the source mask with `self.sourcemask`.
        
        Parameters
        ----------  
        target_srcmask : Mask
            The source mask. 
        sigma : float
            The sigma for the source detection.
        mask_radius_factor : float
            The radius factor for the source detection.
        saturation_level : float
            The saturation level for the source detection.
        save : bool
            If True, save the source mask.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the source mask.
        save_fig : bool
            If True, save the figure of the source mask.
            
        Returns
        -------
        target_sourcemask : Mask
            The source mask.
        """
        from ezphot.methods import MaskGenerator
        maskgenerator = MaskGenerator()
        target_sourcemask = maskgenerator.mask_sources(
            target_img = self,
            target_mask = target_srcmask,
            sigma = sigma,
            mask_radius_factor = mask_radius_factor,
            saturation_level = saturation_level,

            # Others
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig
            )
        return target_sourcemask

    def calculate_bkg(self,
                      target_srcmask: Mask = None,
                      target_ivpmask: Mask = None,
                      is_2D_bkg: bool = True,
                      box_size: int = 64,
                      filter_size: int = 3,
                      save: bool = False,
                      verbose: bool = True,
                      visualize: bool = True,
                      save_fig: bool = False):
        """
        Calculate the background map for this ScienceImage.
        The background map is a map of the background level of the image.
        If save is True, the background map is saved. Then, you can load the background map with `self.bkgmap`.
        
        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        is_2D_bkg : bool
            If True, use 2D background estimation.
        box_size : int
            The box size for the background estimation.
        filter_size : int
            The filter size for the background estimation.
        save : bool
            If True, save the background map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the background map.
        save_fig : bool
            If True, save the figure of the background map.
        
        Returns
        -------
        target_bkg : Background
            The calculated background map.
        """
        from ezphot.methods import BackgroundGenerator
        bkggenerator = BackgroundGenerator()
        target_bkg, _ = bkggenerator.estimate_with_sep(
            target_img = self,
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            is_2D_bkg = is_2D_bkg,
            box_size = box_size,
            filter_size = filter_size,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)

        return target_bkg
    
    def calculate_bkgrms(self, 
                         target_srcmask: Mask = None,
                         target_ivpmask: Mask = None,
                         box_size: int = 64,
                         filter_size: int = 3,
                         save: bool = False,
                         verbose: bool = True,
                         visualize: bool = True,
                         save_fig: bool = False):
        """
        Calculate the background RMS map for this ScienceImage.
        The background RMS map is a map of the background RMS level of the image.
        If save is True, the background RMS map is saved. Then, you can load the background RMS map with `self.bkgrms`.
        
        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        box_size : int
            The box size for the background RMS estimation.
        filter_size : int
            The filter size for the background RMS estimation.
        save : bool
            If True, save the background RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the background RMS map.
        save_fig : bool
            If True, save the figure of the background RMS map.
        
        Returns
        -------
        target_bkgrms : Errormap
            The background RMS map.
        """
        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_bkgrms, _, _ = errormapgenerator.calculate_errormap_from_image(
            target_img = self,  
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            box_size = box_size,
            filter_size = filter_size,
            erormap_type = 'bkgrms',
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return target_bkgrms
    
    def calculate_bkgrms_from_propagation(self,
                                          target_bkg: Background = None,
                                          mbias: CalibrationImage = None,
                                          mdark: CalibrationImage = None,
                                          mflat: CalibrationImage = None,
                                          mflaterr: Errormap = None,
                                          save: bool = False,
                                          verbose: bool = True,
                                          visualize: bool = True,
                                          save_fig: bool = False):
        """
        Calculate the background RMS map for this ScienceImage from the background map, bias frame, dark frame, and flat frame.
        The background RMS map is a map of the background RMS level of the image. 
        If save is True, the background RMS map is saved. Then, you can load the background RMS map with `self.bkgrms`.

        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        mbias : CalibrationImage
            The bias frame.
        mdark : CalibrationImage
            The dark frame.
        mflat : CalibrationImage
            The flat frame.
        mflaterr : Errormap
            The flat error map.
        save : bool
            If True, save the background RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the background RMS map.
        save_fig : bool
            If True, save the figure of the background RMS map.

        Returns
        -------
        target_bkgrms : Errormap
            The background RMS map.
        """
        
        from ezphot.methods import Preprocess
        preprocess = Preprocess()
        # prepare the data
        if target_bkg is None:
            if self.bkgmap is None:
                raise ValueError("Cannot calculate background RMS map: Input background map. OR Register background map with scienceimage.calculate_background(save = True) first.")
            else:
                target_bkg = self.bkgmap
        
        mbias_path, mdark_path, mflat_path = None, None, None
        if mbias is None:
            mbias = preprocess.get_masterframe_from_image(self, 'bias', 30)[0]
        if mdark is None:
            mdark = preprocess.get_masterframe_from_image(self, 'dark', 30)[0]
        if mflat is None:
            mflat = preprocess.get_masterframe_from_image(self, 'flat', 30)[0]
        if mbias is None or mdark is None or mflat is None:
            raise ValueError("Cannot calculate background RMS: required calibration frames are missing.")

        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_bkgrms = errormapgenerator.calculate_bkgrms_from_propagation(
            target_img = self,
            target_bkg = target_bkg,
            mbias_img = mbias,
            mdark_img = mdark,
            mflat_img = mflat,
            mflaterr_img = mflaterr,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
                
        return target_bkgrms
        
    def calculate_errormap(self, 
                           target_srcmask: Mask = None,
                           target_ivpmask: Mask = None,
                           box_size: int = 64,
                           filter_size: int = 3,
                           save: bool = False,
                           verbose: bool = True,
                           visualize: bool = True,
                           save_fig: bool = False):
        """
        Calculate the source RMS map for this ScienceImage.
        The source RMS map is a map of the source RMS level of the image.
        If save is True, the source RMS map is saved. Then, you can load the source RMS map with `self.sourcerms`.

        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        box_size : int
            The box size for the source RMS estimation.
        filter_size : int
            The filter size for the source RMS estimation.
        save : bool
            If True, save the source RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the source RMS map.
        save_fig : bool
            If True, save the figure of the source RMS map.

        Returns
        -------
        target_sourcerms : Errormap
            The source RMS map.
        """
        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_sourcerms = errormapgenerator.calculate_errormap_from_image(
            target_img = self,  
            target_srcmask = target_srcmask,
            target_ivpmask = target_ivpmask,
            box_size = box_size,
            filter_size = filter_size,
            erormap_type = 'sourcerms',
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return target_sourcerms
    
    def calculate_errormap_from_propagation(self,
                                            mbias: CalibrationImage = None,
                                            mdark: CalibrationImage = None,
                                            mflat: CalibrationImage = None,
                                            mflaterr: Errormap = None,
                                            save: bool = False,
                                            verbose: bool = True,
                                            visualize: bool = True,
                                            save_fig: bool = False):
        """
        Calculate the source RMS map for this ScienceImage from the background map, bias frame, dark frame, and flat frame.
        The source RMS map is a map of the source RMS level of the image.
        If save is True, the source RMS map is saved. Then, you can load the source RMS map with `self.sourcerms`.

        Parameters
        ----------
        target_srcmask : Mask
            The source mask.
        target_ivpmask : Mask
            The invalid mask.
        mbias : CalibrationImage
            The bias frame.
        mdark : CalibrationImage
            The dark frame.
        mflat : CalibrationImage
            The flat frame.
        mflaterr : Errormap
            The flat error map.
        save : bool
            If True, save the source RMS map.
        verbose : bool
            If True, print verbose output.
        visualize : bool
            If True, visualize the source RMS map.
        save_fig : bool
            If True, save the figure of the source RMS map.

        Returns
        -------
        target_sourcerms : Errormap
            The source RMS map.
        """
        from ezphot.methods import Preprocess
        preprocess = Preprocess()
        # prepare the data

        mbias_path, mdark_path, mflat_path = None, None, None
        if mbias is None:
            mbias = preprocess.get_masterframe_from_image(self, 'bias', 30)[0]
        if mdark is None:
            mdark = preprocess.get_masterframe_from_image(self, 'dark', 30)[0]
        if mflat is None:
            mflat = preprocess.get_masterframe_from_image(self, 'flat', 30)[0]
        if mbias is None or mdark is None or mflat is None:
            raise ValueError("Cannot calculate source RMS: required calibration frames are missing.")

        from ezphot.methods import ErrormapGenerator
        errormapgenerator = ErrormapGenerator()
        target_sourcerms = errormapgenerator.calculate_sourcerms_from_propagation(
            target_img = self,
            mbias_img = mbias,
            mdark_img = mdark,
            mflat_img = mflat,
            mflaterr_img = mflaterr,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
            
        return target_sourcerms
    
    def get_referenceframe(self, 
                           telname: str = None,
                           min_obsdate: Union[str, float, Time] = None,
                           max_obsdate: Union[str, float, Time] = None,
                           sort_key: Union[str, List[str]] = ['fraction', 'depth'],
                           overlap_threshold: float = 0.5,
                           return_groups: bool = True,
                           group_overlap_threshold: float = 0.8,
                           verbose: bool = True
                           ):
        """
        Get the reference frame from the target image.
        
        Parameters
        ----------
        telname : str, optional
            The telescope name.
        min_obsdate : Union[str, float, Time], optional
            The minimum observation date.
        max_obsdate : Union[str, float, Time], optional
            The maximum observation date.
        sort_key : Union[str, List[str]], optional
            The sort key.
        overlap_threshold : float, optional
            The overlap threshold.
        return_groups : bool, optional
            Whether to return the groups.
        group_overlap_threshold : float, optional
            The group overlap threshold.
            
        Returns
        -------
        reference_img : ReferenceImage
            The reference image.
        reference_frames : Table
            The metadata of the reference frames matched the criteria.
        """
        from ezphot.methods import Subtract
        subtract = Subtract()
        result = subtract.get_referenceframe_from_image(
            target_img = self,
            telname = telname,
            min_obsdate = min_obsdate,
            max_obsdate = max_obsdate,
            sort_key = sort_key,
            overlap_threshold = overlap_threshold,
            return_groups = return_groups,
            group_overlap_threshold = group_overlap_threshold,
            verbose = verbose)
        return result
    
    def get_masterframe(self,
                        imagetyp: str,
                        max_days: float = 10
                        ):
        """
        Get master frame from the image.
        
        This method will search for the master frame in the master frame directory.
        
        Parameters
        ----------
        imagetyp : str
            The type of image to get the master frame from. (BIAS, DARK, FLAT)
        max_days : float, optional
            The maximum number of days to search for the master frame.
            
        Returns
        -------
        master_img : CalibrationImage
            The master frame image.
        master_frames_tbl : Table
            Metadata of the master frame(s) found.
        """
        from ezphot.methods import Preprocess
        preprocess = Preprocess()
        result = preprocess.get_masterframe_from_image(
            target_img = self,
            imagetyp = imagetyp,
            max_days = max_days,
            )
                
        return result
    
    def get_referencecatalog(self,
                             catalog_type: str = 'GAIAXP',
                             fraction_criteria: float = 0.05,
                             query_when_not_archived: bool = True,
                             verbose: bool = False):
        from ezphot.skycatalog import SkyCatalogUtility
        skycatalogutility = SkyCatalogUtility()
        skycatalog = skycatalogutility.get_catalogs(
            ra = self.ra, 
            dec = self.dec, 
            objname = self.objname, 
            fov_ra = self.fov_ra, 
            fov_dec = self.fov_dec, 
            catalog_type = catalog_type, 
            fraction_criteria = fraction_criteria, 
            query_when_not_archived = query_when_not_archived, 
            verbose = verbose)
        return skycatalog
    
    def query_referenceframe(self,
                             save_path: str = None,
                             verbose: bool = True,
                             n_processes: int = 4):
        """
        Query the reference frame from the target image.
        
        Parameters
        ----------
        save_path : str, optional
            The save path of the reference frame.
        verbose : bool, optional
            The verbose flag.
        n_processes : int, optional
            The number of processes.
        
        Returns
        -------
        reference_img : ReferenceImage
            The reference image.
        """
        from ezphot.utils import ImageQuerier
        imagequerier = ImageQuerier()
        result = imagequerier.query(
            width = self.naxis1,
            height = self.naxis2,
            ra = self.center['ra'],
            dec = self.center['dec'],
            pixelscale = self.pixelscale[0],
            telinfo = self.telinfo,
            save_path = save_path,
            objname = self.objname,
            rotation_angle = 0.0,
            verbose = verbose,
            n_processes = n_processes)
        return result
    
    def subtract_background(self, 
                            target_bkg: Background,
                            overwrite: bool = False,
                            save: bool = False,
                            verbose: bool = True,
                            visualize: bool = True,
                            save_fig: bool = False):
        """
        Subtract the background from the target image.
        
        Parameters
        ----------
        target_bkg : ezphot.imageobjects.Background
            The background image to subtract from the target image.
        overwrite : bool, optional
            Whether to overwrite the target image.
        save : bool, optional
            Whether to save the subtracted image.
        verbose : bool, optional
            Whether to print verbose output.
        visualize : bool, optional
            Whether to visualize the subtracted image.
        save_fig : bool, optional
            Whether to save the subtracted image as a figure.
            
        Returns
        -------
        target_img : ScienceImage or ReferenceImage
            The subtracted image.
        """
        
        from ezphot.methods import BackgroundGenerator
        backgroundgenerator = BackgroundGenerator()
        result = backgroundgenerator.subtract_background(
            target_img = self,
            target_bkg = target_bkg,
            overwrite = overwrite,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return result
    
    def photometry_sex(self,
                       target_bkg: Background = None,
                       target_bkgrms: Errormap = None,
                       target_mask: Mask = None,
                       sex_params: dict = None,
                       detection_sigma: float = 5,
                       aperture_diameter_arcsec: Union[float, list] = [5,7,10],
                       aperture_diameter_seeing: Union[float, list] = [3.5,4.5],
                       saturation_level: float = 60000,
                       kron_factor: float = 2.5,
                       save: bool = True,
                       verbose: bool = True,
                       visualize: bool = True,
                       save_fig: bool = False):
        """
        Perform source-extractor photometry on the target image.
        
        Parameters
        ----------
        target_bkg : Background
            The background to subtract from the target image.
        target_bkgrms : Errormap
            The background RMS to use for the photometry.
        target_mask : Mask
            The mask to use for the photometry.
        sex_params : dict
            The source-extractor parameters.
        detection_sigma : float
            The detection sigma for the photometry.
        aperture_diameter_arcsec : Union[float, list]
            The aperture diameter in arcseconds.
        aperture_diameter_seeing : Union[float, list]
            The aperture diameter in seeing units.
        saturation_level : float
            The saturation level for the photometry.
        kron_factor : float
            The Kron factor for the photometry.
        save : bool
            Whether to save the photometry catalog.
        verbose : bool
            Whether to print verbose output.
        visualize : bool
            Whether to visualize the photometry.
        save_fig : bool
            Whether to save the photometry figure.

        Returns
        -------
        result : Catalog
            The photometry catalog.
        """
        
        from ezphot.methods import AperturePhotometry
        aperturephotometry = AperturePhotometry()
        result = aperturephotometry.sex_photometry(
            target_img = self,
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            target_mask = target_mask,
            sex_params = sex_params,
            detection_sigma = detection_sigma,
            aperture_diameter_arcsec = aperture_diameter_arcsec,
            aperture_diameter_seeing = aperture_diameter_seeing,
            saturation_level = saturation_level,
            kron_factor = kron_factor,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return result
    
    def photometry_forced_circular(self,
                                   x_arr: list = None,
                                   y_arr: list = None,
                                   aperture_diameter_arcsec: Union[float, list] = [5,7,10],
                                   aperture_diameter_seeing: Union[float, list] = [3.5,4.5],
                                   annulus_width_arcsec: Union[float, list] = None,
                                   unit: str = 'pixel',
                                   target_bkg: Background = None,
                                   target_bkgrms: Errormap = None,
                                   target_mask: Mask = None,
                                   
                                   save: bool = True,
                                   verbose: bool = True,
                                   visualize: bool = True,
                                   save_fig: bool = False):
        '''
        Perform forced photometry on the target image.
        
        Parameters
        ----------
        x_arr : list
            The x-coordinates of the sources.
        y_arr : list
            The y-coordinates of the sources.
        aperture_diameter_arcsec : Union[float, list]
            The aperture diameter in arcseconds.
        aperture_diameter_seeing : Union[float, list]
            The aperture diameter in seeing units.
        annulus_width_arcsec : Union[float, list]
            The annulus width in arcseconds.
        unit : str
            The unit of the coordinates.
        target_bkg : Background
            The background to subtract from the target image.
        target_bkgrms : Errormap
            The background RMS to use for the photometry.
        target_mask : Mask
            The mask to use for the photometry.
        save : bool
            Whether to save the photometry catalog.
        verbose : bool
            Whether to print verbose output.
        visualize : bool
            Whether to visualize the photometry.
        save_fig : bool
            Whether to save the photometry figure.

        Returns
        -------
        result : Catalog
            The photometry catalog.
        '''

        from ezphot.methods import AperturePhotometry
        aperturephotometry = AperturePhotometry()
        result = aperturephotometry.circular_photometry(
            target_img = self,
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            target_mask = target_mask,
            x_arr = x_arr,
            y_arr = y_arr,
            aperture_diameter_arcsec = aperture_diameter_arcsec,
            aperture_diameter_seeing = aperture_diameter_seeing,
            annulus_width_arcsec = annulus_width_arcsec,
            unit = unit,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return result
        
    def photometry_forced_elliptical(self,
                                     x_arr: list = None,
                                     y_arr: list = None,
                                     sma_arr: list = None,
                                     smi_arr: list = None,
                                     theta_arr: list = None,
                                     unit: str = 'pixel',
                                     annulus_ratio: float = None,
                                     target_bkg: Background = None,
                                     target_bkgrms: Errormap = None,
                                     target_mask: Mask = None,
                                     
                                     save: bool = True,
                                     verbose: bool = True,
                                     visualize: bool = True,
                                     save_fig: bool = False):
        '''
        Perform forced photometry on the target image.
        
        Parameters
        ----------
        x_arr : list
            The x-coordinates of the sources.
        y_arr : list
            The y-coordinates of the sources.
        sma_arr : list
            The semi-major axis of the sources.
        smi_arr : list
            The semi-minor axis of the sources.
        theta_arr : list
            The position angle of the sources.
        unit : str
            The unit of the coordinates.
        annulus_ratio : float
            The ratio of the annulus diameter to the semi-major axis.
        target_bkg : Background
            The background to subtract from the target image.
        target_bkgrms : Errormap
            The background RMS to use for the photometry.
        target_mask : Mask
            The mask to use for the photometry.
        save : bool
            Whether to save the photometry catalog.
        verbose : bool
            Whether to print verbose output.
        visualize : bool
            Whether to visualize the photometry.
        save_fig : bool
            Whether to save the photometry figure.

        Returns
        -------
        result : Catalog
            The photometry catalog.
        '''
        
        from ezphot.methods import AperturePhotometry
        aperturephotometry = AperturePhotometry()
        result = aperturephotometry.elliptical_photometry(
            target_img = self,
            target_bkg = target_bkg,
            target_bkgrms = target_bkgrms,
            target_mask = target_mask,
            x_arr = x_arr,
            y_arr = y_arr,
            sma_arr = sma_arr,
            smi_arr = smi_arr,
            theta_arr = theta_arr,
            unit = unit,
            annulus_ratio = annulus_ratio,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig)
        return result
        
    def photometric_calibration(self, 
                                target_catalog,
                                catalog_type: str = 'GAIAXP',
                                max_distance_second: float = 1.0,
                                calculate_color_terms: bool = True,
                                calculate_mag_terms: bool = True,
                                
                                # Selection parameters
                                mag_lower: float = 13,
                                mag_upper: float = 15,
                                dynamic_mag_range: bool = True,
                                classstar_lower: float = 0.8,
                                elongation_upper: float = 1.7,
                                elongation_sigma: float = 5,
                                fwhm_lower: float = 1,
                                fwhm_upper: float = 15,
                                fwhm_sigma: float = 5,
                                flag_upper: int = 1,
                                maskflag_upper: int = 1,
                                inner_fraction: float = 0.7, # Fraction of the images
                                isolation_radius: float = 10.0,
                                magnitude_key: str = 'MAG_AUTO',
                                fwhm_key: str = 'FWHM_IMAGE',
                                x_key: str = 'X_IMAGE',
                                y_key: str = 'Y_IMAGE',
                                classstar_key: str = 'CLASS_STAR',
                                elongation_key: str = 'ELONGATION',
                                flag_key: str = 'FLAGS',
                                maskflag_key: str = 'IMAFLAGS_ISO',

                                # Other parameters
                                update_header: bool = True,
                                save: bool = True,
                                verbose: bool = True,
                                visualize: bool = True,
                                save_fig: bool = False,
                                save_refcat: bool = True
                                ):
        
        from ezphot.methods import PhotometricCalibration
        from ezphot.dataobjects import Catalog
        if not isinstance(target_catalog, Catalog):
            raise ValueError("target_catalog must be a Catalog object.")
        photometriccalibration = PhotometricCalibration()
    
        result = photometriccalibration.photometric_calibration(
            target_img = self,
            target_catalog = target_catalog,
            catalog_type = catalog_type,
            max_distance_second = max_distance_second,
            calculate_color_terms = calculate_color_terms,
            calculate_mag_terms = calculate_mag_terms,
            mag_lower = mag_lower,
            mag_upper = mag_upper,
            dynamic_mag_range = dynamic_mag_range,
            classstar_lower = classstar_lower,
            elongation_upper = elongation_upper,
            elongation_sigma = elongation_sigma,
            fwhm_lower = fwhm_lower,
            fwhm_upper = fwhm_upper,
            fwhm_sigma = fwhm_sigma,
            flag_upper = flag_upper,
            maskflag_upper = maskflag_upper,
            inner_fraction = inner_fraction,
            isolation_radius = isolation_radius,
            magnitude_key = magnitude_key,
            fwhm_key = fwhm_key,
            x_key = x_key,
            y_key = y_key,
            classstar_key = classstar_key,
            elongation_key = elongation_key,
            flag_key = flag_key,
            maskflag_key = maskflag_key,
            update_header = update_header,
            save = save,
            verbose = verbose,
            visualize = visualize,
            save_fig = save_fig,
            save_refcat = save_refcat)
        
        return result

    def reproject(self, 
                  target_errormap: Optional[Errormap] = None,
                  swarp_params: Optional[dict] = None,
                  resample_type: str = 'LANCZOS3',
                  center_ra: Optional[float] = None,
                  center_dec: Optional[float] = None,
                  x_size: Optional[int] = None,
                  y_size: Optional[int] = None,
                  pixelscale: Optional[float] = None,
                  verbose: bool = True,
                  overwrite: bool = False,
                  save: bool = True,
                  return_ivpmask: bool = False,
                  fill_zero_tonan: bool = True):
        """
        Reproject the image with SWarp.
        
        Parameters
        ----------
        target_errormap : Errormap
            The error map to use for the reproject.
        swarp_params : dict
            The parameters for SWarp.
        resample_type : str
            The type of resampling for SWarp.
        center_ra : float
            The center RA for SWarp.
        center_dec : float
            The center Dec for SWarp.
        x_size : int
            The size of the image in the x direction for SWarp.
        y_size : int
            The size of the image in the y direction for SWarp.
        pixelscale : float
            The pixel scale for SWarp.
        verbose : bool
            Whether to print verbose output.
        overwrite : bool
            Whether to overwrite the existing reprojected image.
        save : bool
            Whether to save the reprojected image.
        return_ivpmask : bool
            Whether to return the invalid pixel mask.
        fill_zero_tonan : bool
            Whether to fill the zero to nan.
        """
        from ezphot.methods import Reproject
        reproject = Reproject()
        result = reproject.reproject(
            target_img = self,
            target_errormap = target_errormap,
            swarp_params = swarp_params,
            resample_type = resample_type,
            center_ra = center_ra,
            center_dec = center_dec,
            x_size = x_size,
            y_size = y_size,
            pixelscale = pixelscale,
            verbose = verbose,
            overwrite = overwrite,
            save = save,
            return_ivpmask = return_ivpmask,
            fill_zero_tonan = fill_zero_tonan)
        return result
    
    def DIA(self,
            reference_img: "ReferenceImage" = None,
            target_bkg: Background = None,
            
            # Photometry configuration
            detection_sigma: float = 1.5,
            aperture_diameter_arcsec: List[float] = [5, 7, 10],
            aperture_diameter_seeing: List[float] = [3.5, 4.5],
            kron_factor: float = 1.5,
            catalog_type: str = 'GAIAXP',
            
            # DIA configuration
            reject_variable_sources: bool = False,
            negative_detection: bool = True,
            reverse_subtraction: bool = False,            
            save_transient_figure: bool = True,
            save_candidate_figure: bool = True,
            show_transient_numbers: int = 100,
            show_candidate_numbers: int = 100,
            # HOTPANTS parameters
            convim: str = 't',
            normim: str = 'i',
            tu: float = None,
            tl: float = None,
            iu: float = None,
            il: float = None,
            nrx: int = 1,
            nry: int = 1,
            nsx: int = 10,
            nsy: int = 10,
            ko: int = 2,
            bgo: int = 1,
            r: int = 10,
            
            # Other parameters
            save: bool = True,
            verbose: bool = True,
            visualize: bool = False
            ):
        
        from ezphot.methods import Subtract
        if reference_img is None:
            reference_result = self.get_referenceframe(verbose = verbose)
            if reference_result is None:
                raise RuntimeError("No reference image found.")            
            reference_img = reference_result[0]
        
        reference_img = np.atleast_1d(reference_img)
        
        subtract = Subtract()
        result = subtract.find_transients(
            target_img = self,
            reference_imglist = reference_img,
            target_bkg = target_bkg,
            detection_sigma = detection_sigma,
            aperture_diameter_arcsec = aperture_diameter_arcsec,
            aperture_diameter_seeing = aperture_diameter_seeing,
            kron_factor = kron_factor,
            catalog_type = catalog_type,
            reject_variable_sources = reject_variable_sources,
            negative_detection = negative_detection,
            reverse_subtraction = reverse_subtraction,
            save_transient_figure = save_transient_figure,
            save_candidate_figure = save_candidate_figure,
            show_transient_numbers = show_transient_numbers,
            show_candidate_numbers = show_candidate_numbers,
            convim = convim,
            normim = normim,
            tu = tu,
            tl = tl,
            iu = iu,
            il = il,
            nrx = nrx,
            nry = nry,
            nsx = nsx,
            nsy = nsy,
            ko = ko,
            bgo = bgo,
            r = r,
            save = save,
            verbose = verbose,
            visualize = visualize,
        )
        return result
# %%
