#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:33:40 2022

@author: hhchoi1022
"""

# %%

# Photometry for a designated point source 

# Creating Aperture Objects from RADec
from photutils.aperture import CircularAperture
import astropy.units as u
from astropy.io import ascii
import os
os.chdir('/home/hhchoi1022/Desktop/Gitrepo/makereference')
from image_evaluation import UPDATE_IMAGE
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from datetime import datetime
import matplotlib.pyplot as plt
import re
from astropy.io import fits
import glob
import sys
#%%

class Photometry:
    
    def __init__(self, imkey = None, refimage = None, obsinfo = None, obj_ra = None, obj_dec = None):

        
        if type(imkey) == str:
            self.__imlist = sorted(glob.glob(imkey))
            self.__field = fits.getheader(self.__imlist[0])['object']
        self.__obsinfo = obsinfo
        self.__refim = refimage
        self.__ra = obj_ra
        self.__dec = obj_dec
        self.__exptime = fits.getheader(self.__imlist[0])['EXPTIME']
    
    def get_imlist(self):
        return self.__imlist

    def get_obsinfo(self):
        return self.__obsinfo

    def get_field(self):
        return self.__field

    def get_mode(self):
        return self.__mode

    def get_targetsize(self):
        
        IMSNG_fieldlist =  ascii.read('/home/hhchoi1022/Desktop/Gitrepo/config/alltarget.dat')
        
        pixsize = self.__obsinfo['pixelscale']
        
        targetsize = IMSNG_fieldlist[IMSNG_fieldlist['obj'] == self.__field]['maxaxis']*60/pixsize
        if targetsize <=64:
            bkgsize = 64
        elif targetsize > 64 and targetsize <=128:
            bkgsize = 128
        elif targetsize >128 and targetsize <=400:
            bkgsize = 256  
        elif targetsize >400:
            bkgsize = 512
            
        self.__bkgsize = bkgsize
        
        return self.__bkgsize

    def get_radec(self):
        # Input : RADec of the target object
        # Outpout : Position in SkyCoord format
        from astropy.coordinates import SkyCoord
        
        if type(self.__ra) != str:
            position = SkyCoord(self.__ra, self.__dec, unit = u.deg, frame ='fk5')
        elif type(self.__ra) == str:
            position = SkyCoord(self.__ra, self.__dec, unit = (u.hourangle,u.deg), frame = 'fk5')
        self.__radec = position
        return self.__radec

    def update_refim(self, refim):
        self.__refim = refim

    def update_radec(self, detectiondate='20210101', check = True):
        
        detection_date = datetime.strptime(detectiondate,'%Y%m%d')
        imlist = self.__imlist
        
        detimlist = []
        for image in imlist:
            date = re.findall('(20\d\d\d\d\d\d)', os.path.basename(image))[0]
            obsdate = datetime.strptime(date,'%Y%m%d')
            if obsdate > detection_date:
                detimlist.append(image)
                
        positionlist_ra = []
        positionlist_dec = []
        noinfo = []
        for image in detimlist:
            try:
                cutout_hdu, _=self.cutout(image, write = True)
                hdr = cutout_hdu.header 
                wcs = WCS(hdr)
                position_g = self.get_detection_DAOfind(cutout_hdu, check = True)
                position_g_x = position_g[np.argmax(position_g['peak'])]['xcentroid']
                position_g_y = position_g[np.argmax(position_g['peak'])]['ycentroid']
                coord_ra = SkyCoord.from_pixel(position_g_x,position_g_y,wcs=wcs).ra.degree
                coord_dec = SkyCoord.from_pixel(position_g_x,position_g_y,wcs=wcs).dec.degree
                positionlist_ra.append(coord_ra)
                positionlist_dec.append(coord_dec)
            except:
                noinfo.append(image)
        ra_guess = np.median(sigma_clip(positionlist_ra,sigma=3,maxiters=1))
        dec_guess =  np.median(sigma_clip(positionlist_dec,sigma=3,maxiters=1))
        wcs = WCS(fits.getheader(image))
        if check == True:
            fig = plt.figure(figsize = (5,5))
            ax = fig.add_subplot(111, projection = wcs)
            plt.xlabel('RA')
            plt.ylabel('Dec')
            plt.scatter(positionlist_ra, positionlist_dec, s = 10, marker = '.', c = 'k',transform = ax.get_transform('world'))
            plt.scatter(ra_guess, dec_guess, s = 20, marker = '+', c= 'r', transform = ax.get_transform('world'))
            plt.grid()
            plt.show()
            
        self.__ra = float(ra_guess)
        self.__dec = float(dec_guess)
        
        return self.__ra, self.__dec

    def select_outlier(self, remove = False, check = True):
        from astropy.stats import sigma_clip
        from astropy.table import Table
        from astropy.io import fits
        import matplotlib.pyplot as plt
        
        seeinglist = []
        depthlist = []
        imagelist = []
        
        for image in self.__imlist:
            hdr = fits.getheader(image)
            try:
                depth = hdr['UL5_4']
                seeing = hdr['SEEING']
                seeinglist.append(seeing)
                depthlist.append(depth)
                imagelist.append(image)
            except:
                pass
        
        all_tbl = Table()
        all_tbl['Image'] = imagelist
        all_tbl['Depth'] = depthlist
        all_tbl['Seeing'] = seeinglist
        
        cut_tbl = all_tbl[(sigma_clip(all_tbl['Depth'],sigma = 3).mask) | (sigma_clip(all_tbl['Seeing'],sigma = 3).mask)]
        selected_tbl = all_tbl[~(sigma_clip(all_tbl['Depth'],sigma = 3).mask) & (~sigma_clip(all_tbl['Seeing'],sigma = 3).mask)]
        
        if check == True:
            plt.figure(figsize = (10,5))
            plt.scatter(all_tbl['Seeing'],all_tbl['Depth'], c= 'k', marker = 'o', alpha = 0.6)
            plt.scatter(cut_tbl['Seeing'],cut_tbl['Depth'], c= 'r', label = 'Outlier', marker = 'o', alpha = 0.6)
            plt.xlabel('Seeing[arcsec]')
            plt.ylabel('Depth[AB]')
            plt.grid()
            plt.legend()
            plt.show()
        if (remove == True) & (len(cut_tbl) > 0):
                os.system(f'rm {image}')
                self.__imlist = selected_tbl['Image'].tolist()
        
        return selected_tbl, cut_tbl

    def divide_RASA(self):
        import os, glob
        from astropy.io import fits
        
        for i, image in enumerate(self.__imlist):
            self.printProgress(i,len(self.__imlist),prefix = 'Progress:',suffix = ' ')
            imdir = os.path.dirname(image)
            hdr = fits.getheader(image)
            if 'CAMMODE' in hdr.keys():
                if hdr['CAMMODE'] == 'HIGH':
                    os.makedirs(f'{imdir}/HIGH', exist_ok = True)
                    os.system(f'cp {image} {imdir}/HIGH')
                    print(f'{image} is sorted as HIGH')
                else:
                    os.makedirs(f'{imdir}/MERGE', exist_ok = True)
                    os.system(f'cp {image} {imdir}/MERGE')
                    print(f'{image} is sorted as MERGE')
            else:
                data = fits.getdata(image)
                zero = np.mean(np.sort((data[~np.isnan(data)].flatten()))[-10:-5])
                if zero > 80000:
                    os.makedirs(f'{imdir}/MERGE', exist_ok = True)
                    os.system(f'mv {image} {os.path.dirname(image)}/MERGE')
                    print(f'{image} is sorted as MERGE')
                elif zero < 80000:
                    os.makedirs(f'{imdir}/HIGH', exist_ok = True)
                    os.system(f'mv {image} {os.path.dirname(image)}/HIGH')
                    print(f'{image} is sorted as HIGH')
                    
    def cutout(self, image, factor_size = 5, cutoutsize = None, write=False):
        from astropy.nddata import Cutout2D
        import os
        from astropy.wcs import WCS
        from astropy.io import fits
        
        #Getting data
        if type(image) == str:
            hdu = fits.open(image)[0]
        else :
            hdu = image
        
        #Cutting out data
        wcs = WCS(hdu.header)
        best_aperture = float(3* hdu.header['SEEING']/self.__obsinfo['pixelscale'])
        position = self.get_radec()
        position_pixel = position.to_pixel(wcs)
        size = float(best_aperture*factor_size)
        if cutoutsize != None:
            size = cutoutsize
        cutouted = Cutout2D(hdu.data,position_pixel, size, wcs= wcs)
        hdu.data = cutouted.data
        hdu.header.update(cutouted.wcs.to_header())
        curpath = os.path.dirname(image)
        cutout_hdu = hdu
        cutout_image_path = ''
        if write == True:
            os.makedirs(f'{curpath}/cutout', exist_ok = True)
            os.chdir(f'{curpath}/cutout')
            cutout_hdu.writeto(f'{curpath}/cutout/{os.path.basename(image)}',overwrite=True)
            cutout_image_path = f'{curpath}/cutout/{os.path.basename(image)}'
            return cutout_hdu, cutout_image_path
        else:
            return cutout_hdu, cutout_image_path
        
    def calc_jdmean(self, imlist):
        from astropy.time import Time
        import numpy as np
        from astropy.io import fits
        
        if type(imlist[0]) == str:
            average_time = Time(np.mean([fits.getheader(inim)['JD'] for inim in imlist]), format='jd')
        else:
            average_time = Time(np.mean([hdu[0].header['JD'] for hdu in imlist]), format='jd')
        
        return average_time

    def calc_totexp(self, imlist):
        from astropy.io import fits
        import numpy as np
        
        if type(imlist[0]) == str:
            totexp = int(np.sum([fits.getheader(inim)['EXPTIME'] for inim in imlist]))
        else:
            totexp = int(np.sum([hdu[0].header['EXPTIME'] for hdu in imlist]))
        
        return totexp

    def combine_ccdproc(self, alignedlist, write = False):
        from astropy.nddata import CCDData
        from astropy.io import fits
        import astropy.units as u
        from ccdproc import Combiner
        
        # Constructing data to combine
        comdataarray = []
        if type(alignedlist[0]) == str:
            for image in alignedlist:
                hdu = fits.open(image)
                data = CCDData(hdu[0].data, unit = u.adu)
                comdataarray.append(data)
        else:
            for hdu in alignedlist:
                data = CCDData(hdu[0].data, unit = u.adu)
                comdataarray.append(data)
        hdr = hdu[0].header
        jd = self.calc_jdmean(alignedlist)
        totexp = self.calc_totexp(alignedlist)
            
        combiner = Combiner(comdataarray)
        if len(comdataarray) > 3:
            combiner.clip_extrema(nhigh=1)
        hdu = combiner.median_combine()
        
        # Constructing header
        hdr['JD'] = (jd.value, 'Julian Date')
        hdr['MJD'] = (jd.mjd, 'Modified Julian Date at start of exposure')
        hdr['NCOMBINE'] = len(alignedlist)
        hdr['EXPTIME'] = (totexp, 'Total Exposure time [sec]')
        hdr['DATE-OBS'] = (jd.isot)
        
        combined_hdu = fits.PrimaryHDU(data = hdu.data, header = hdr)
        combined_hdu = fits.HDUList([combined_hdu])
        
        # Writing combined output
        if write == True:
            fits.writeto(f'{self.__field}_{self.__obsinfo["obs"][0].strip()}_{round(jd.mjd,4)}_{totexp}.com.fits', combined_hdu[0].data, combined_hdu[0].header, overwrite=True)
            return f'{self.__field}_{self.__obsinfo["obs"][0].strip()}_{round(jd.mjd,4)}_{totexp}.com.fits'
        else:
            return combined_hdu
    
    def combine_iraf(self, alignedlist):
        import os
        os.chdir('/data2/iraf')
        from pyraf import iraf
        
        imdir = os.path.dirname(alignedlist[0])
        n_im = len(alignedlist)
        jd = self.calc_jdmean(alignedlist)
        totexp = self.calc_totexp(alignedlist)

        
        os.chdir(imdir)
        iraf.chdir(imdir)
        with open('image.list','w',encoding='UTF-8') as f:
            for name in alignedlist:
                f.write(name+'\n')
                
        if not n_im == 0:
            if n_im > 30:
                iraf.imcombine('@image.list', f'{self.__field}_{self.__obsinfo["obs"][0].strip()}_{totexp}.ref.fits', combine = 'median', reject = 'minmax', zero = 'mode',nlow = 2, nhigh= 4)
            elif n_im > 15:
                iraf.imcombine('@image.list', f'{self.__field}_{self.__obsinfo["obs"][0].strip()}_{totexp}.ref.fits', combine = 'median', reject = 'minmax', zero = 'mode',nlow = 1, nhigh= 3)
            else:
                iraf.imcombine('@image.list', f'{self.__field}_{self.__obsinfo["obs"][0].strip()}_{totexp}.ref.fits', combine = 'median', reject = 'minmax', zero = 'mode', nhigh= 1)

            os.system('rm image.list') 
        return f'{imdir}/{self.__field}_{self.__obsinfo["obs"][0].strip()}_{totexp}.ref.fits'

    def scale_bkg(self, image):
        from astropy.io import fits
        import numpy as np
        
        tgtim_hdu = fits.open(self.__refim)
        image_hdu = fits.open(image)
        
        if ('SKYVAL' in image_hdu[0].header.keys()) & ('SKYVAL' in tgtim_hdu[0].header.keys()):
            bkg_offset = image_hdu[0].header['SKYVAL'] - tgtim_hdu[0].header['SKYVAL']
            
        else:
            # Referenced by Gregory Paek
            tgt_bkg = np.median(tgtim_hdu[0].data[~np.isnan(tgtim_hdu[0].data)].flatten())
            img_bkg = np.median(image_hdu[0].data[~np.isnan(image_hdu[0].data)].flatten())
            bkg_offset = tgt_bkg-img_bkg
            
        scaled_data = image_hdu[0].data-bkg_offset
        image_hdu[0].data = scaled_data
        
        return image_hdu

    def align(self, image, scale = True, write = False):
        from astropy.io import fits
        import astroalign as aa
        from astropy.wcs import WCS
        
        ref_hdu = fits.open(self.__refim)
        ref_data = ref_hdu[0].data
        ref_hdr = ref_hdu[0].header
        
        image_hdu = fits.open(image)
        image_data = image_hdu[0].data
        image_data = image_data.byteswap().newbyteorder()
        if scale == True:
            image_data = self.scale_bkg(image)[0].data
        image_hdr = image_hdu[0].header
        for key in ['JD','DATE-OBS','JD-OBS','MJD','SEEING','PEEING','APER_3','ZP_4','UL5_4']:
            if (key in ref_hdr.keys()) & (key in image_hdr.keys()):
                ref_hdr[key] = image_hdr[key]
        
        aligned_data, footprint = aa.register(image_data, ref_data, fill_value = 0)
        
        aligned_hdu = fits.PrimaryHDU(data = aligned_data, header = ref_hdr)
        aligned_hdu = fits.HDUList([aligned_hdu])
        imdir = os.path.dirname(image)
        if write == True:
            os.makedirs(f'{imdir}/aligned', exist_ok=True)
            os.chdir(f'{imdir}/aligned')
            fits.writeto(f'{imdir}/aligned/aligned_{os.path.basename(image)}', aligned_hdu[0].data, aligned_hdu[0].header, overwrite = True)
        aligned_image = f'{imdir}/aligned/aligned_{os.path.basename(image)}'
        return aligned_hdu, aligned_image

    def subtraction_hotpants(self, image, refim, iu = 60000, tu = 6000000000, tl = -100000, v = 0, ng = '3 3 1.0 2 0.7 1 0.4'):
        imdir = os.path.dirname(image)
        os.system(f'hotpants -c t -n i -inim {image} -tmplim {refim} -outim {imdir}/sub_{os.path.basename(image)} -iu {iu} -tu {tu} -tl {tl} -v {v} -ng {ng}')

    def get_detection_DAOfind(self, cutout_image, sigmafactor = 3.0, threshold = 5, check = False):
        from photutils.detection import DAOStarFinder
        from astropy.stats import sigma_clipped_stats
        from astropy.io import fits
        import matplotlib.pyplot as plt
        import numpy as np
        
        #Getting data
        if type(cutout_image) == str:
            cutout_hdu = fits.open(cutout_image)[0]
        else:
            cutout_hdu = cutout_image
        data = cutout_hdu.data
        hdr = cutout_hdu.header
        
        #Detecting source
        aperture = float(1.5*hdr['SEEING']/self.__obsinfo['pixelscale'])
        mean, median, std = sigma_clipped_stats(data, sigma = sigmafactor )
        daofind = DAOStarFinder( threshold = threshold*std , fwhm = aperture)
        sources = daofind(data-median)
        if sources != None:
            sources = sources[(abs(sources['xcentroid']-len(data)/2)<15)&(abs(sources['ycentroid']-len(data)/2)<15)]
        
        if check == True:
            plt.figure()
            plt.imshow(data,origin='lower',interpolation='nearest')
            xcent = sources[np.argmax(sources['flux'])]['xcentroid']
            ycent = sources[np.argmax(sources['flux'])]['ycentroid']
            plt.scatter(xcent, ycent, s = 10, marker = '+', c ='r', label = 'detection')
            plt.colorbar()
            plt.legend()
            plt.show()
            
        return sources

    def photometry_photutils(self, image, factor_aperture = 3, zpkey = 'ZP_4', threshold = 5, check = False):
        import numpy as np
        from photutils.aperture import aperture_photometry
        from astropy.visualization import simple_norm
        import matplotlib.pyplot as plt
        from photutils.aperture import CircularAnnulus
        from astropy.stats import sigma_clipped_stats
        from astropy.wcs import WCS
        
        # Getting data
        cutout_hdu, cutout_image = self.cutout(image, factor_size = 8, write = True)
        data = cutout_hdu.data
        hdr = cutout_hdu.header 
        zp = hdr[zpkey]
        wcs = WCS(hdr)
        
        # Setting the aperture 
        aper_radius = float(factor_aperture/2*hdr['SEEING']/float(self.__obsinfo['pixelscale']))
        detection = self.get_detection_DAOfind(cutout_hdu, sigmafactor = 3, threshold = 5)

        if detection == None:
            position_pixel = self.get_radec().to_pixel(wcs = wcs)
            color = 'black'
        elif len(detection) == 0:
            position_pixel = self.get_radec().to_pixel(wcs = wcs)
            color = 'black'
        else:
            detection = detection[np.argmax(detection['flux'])]
            position_pixel = float(detection['xcentroid']), float(detection['ycentroid'])
            color = 'red'
        aperture = CircularAperture(position_pixel,aper_radius)
        
        # Local sigma_clipped background estimation
        annulus_aperture = CircularAnnulus(position_pixel, r_in = 2*aper_radius, r_out = 3*aper_radius)
        annulus_mask = annulus_aperture.to_mask(method = 'center')
        annulus_data = annulus_mask.multiply(data)
        mask = annulus_mask.data
        annulus_data_1d = annulus_data[mask>0]
        _, bkg_median, bkg_std = sigma_clipped_stats(annulus_data_1d)
    
        # Checking aperture 
        if check == True:
            plt.figure(figsize= (8,2.5))
            plt.subplot(1,2,1)
            norm = simple_norm(data, 'sqrt', percent = 99)
            plt.imshow(data,origin='lower',interpolation='nearest', norm = norm)
            ann_patches = annulus_aperture.plot(color = color, lw = 2, label = 'Background Annulus')
            ap_patches = aperture.plot(color = 'white', lw = 2, label = 'Aperture')
            handles = (ap_patches[0], ann_patches[0])
            plt.legend(loc = (0.17,0.05), facecolor = '#458989', labelcolor = 'white', handles = handles, prop = {'size':6})
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(annulus_data,origin='lower', interpolation='nearest')
            plt.colorbar()
    
        # Error estimation
        subtracted_data = data - bkg_median
        if 'NCOMBINE' in hdr.keys():    
            noise_read = float(self.__obsinfo['readnoise'])/np.sqrt(hdr['NCOMBINE'])
            noise_dark = float(np.sqrt(self.__obsinfo['dark']*self.__exptime))/np.sqrt(hdr['NCOMBINE'])
            noise_sky = float(np.sqrt(hdr['SKYVAL']))/np.sqrt(hdr['NCOMBINE'])
        else:
            noise_read = float(self.__obsinfo['readnoise'])
            noise_dark = float(np.sqrt(self.__obsinfo['dark']*self.__exptime))
            noise_sky = float(np.sqrt(hdr['SKYVAL']))
        if not noise_dark > 0:
            noise_dark = 0

        gain = float(self.__obsinfo['gain'])
        noise_subtot = np.sqrt(noise_read**2+noise_sky**2)*np.ones([len(subtracted_data),len(subtracted_data)]) ############################# CHANGED ###############
          
        # Performing photometry
        from photutils.utils import calc_total_error
        noise_tot = calc_total_error(subtracted_data, noise_subtot, gain)
        phot_table = aperture_photometry(subtracted_data, aperture, error = noise_tot)
    
        mag = -2.5*np.log10(phot_table[0]['aperture_sum']) + zp
        magerr = 2.5*phot_table[0]['aperture_sum_err']/2.303/phot_table[0]['aperture_sum']
        if (mag > hdr['UL5_4']) | (phot_table['aperture_sum']<0):
            return hdr['UL5_4'], hdr['ZPER_4']
        return mag, magerr

    def photometry_SE(self, image, factor_aperture = 3, depthkey = 'UL5_4', zpkey = 'ZP_4'):
        import os
        from astropy.io import ascii
        from astropy.io import fits
        import numpy as np
        from astropy.wcs import WCS
        
        conf_param = dict(
            # Default configuration file for Source Extractor 2.25.0
            # EB 2018-02-08
            #
             
            #-------------------------------- Catalog ------------------------------------
             
            CATALOG_NAME     ='photometry.cat',       # name of the output catalog
            CATALOG_TYPE     ='ASCII_HEAD',     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,
                                            # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC
            PARAMETERS_NAME  ='photometry.SEparam',  # name of the file containing catalog contents
             
            #------------------------------- Extraction ----------------------------------
             
            DETECT_TYPE      ='CCD',            # CCD (linear) or PHOTO (with gamma correction)
            DETECT_MINAREA   =5,              # min. # of pixels above threshold
            DETECT_MAXAREA   =0,              # max. # of pixels above threshold (0=unlimited)
            THRESH_TYPE      ='RELATIVE',       # threshold type: RELATIVE (in sigmas)
                                            # or ABSOLUTE (in ADUs)
            DETECT_THRESH    =1.5,            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
            ANALYSIS_THRESH  =1.5,            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2
             
            FILTER           ='Y',              # apply filter for detection (Y or N)?
            FILTER_NAME      ='default.conv',   # name of the file containing the filter
             
            DEBLEND_NTHRESH  =32,             # Number of deblending sub-thresholds
            DEBLEND_MINCONT  =0.005,          # Minimum contrast parameter for deblending
             
            CLEAN            ='Y',              # Clean spurious detections? (Y or N)?
            CLEAN_PARAM      =1.0,            # Cleaning efficiency
             
            MASK_TYPE        ='CORRECT',        # type of detection MASKing: can be one of
                                            # NONE, BLANK or CORRECT
             
            #-------------------------------- WEIGHTing ----------------------------------
            
            WEIGHT_TYPE      ='NONE',           # type of WEIGHTing: NONE, BACKGROUND,
                                            # MAP_RMS, MAP_VAR or MAP_WEIGHT
            RESCALE_WEIGHTS  ='Y',              # Rescale input weights/variances (Y/N)?
            WEIGHT_IMAGE     ='weight.fits',    # weight-map filename
            WEIGHT_GAIN      ='Y',              # modulate gain (E/ADU) with weights? (Y/N)
            
            #------------------------------ Photometry -----------------------------------
             
            PHOT_APERTURES   ='5',              # MAG_APER aperture diameter(s) in pixels
            PHOT_AUTOPARAMS  ='2.5,3.5',       # MAG_AUTO parameters: <Kron_fact>,<min_radius>
            PHOT_PETROPARAMS ='2.0,3.5',       # MAG_PETRO parameters: <Petrosian_fact>,
                                            # <min_radius>
            PHOT_AUTOAPERS   ='0.0,0.0',        # <estimation>,<measurement> minimum apertures
                                            # for MAG_AUTO and MAG_PETRO
            PHOT_FLUXFRAC    =0.5,            # flux fraction[s] used for FLUX_RADIUS
             
            SATUR_LEVEL      =50000.0,        # level (in ADUs) at which arises saturation
            SATUR_KEY        ='SATURATE',       # keyword for saturation level (in ADUs)
             
            MAG_ZEROPOINT    =0.0,            # magnitude zero-point
            MAG_GAMMA        =4.0,            # gamma of emulsion (for photographic scans)
            GAIN             =0.0,            # detector gain in e-/ADU
            GAIN_KEY         ='GAIN',           # keyword for detector gain in e-/ADU
            PIXEL_SCALE      =1.0,            # size of pixel in arcsec (0=use FITS WCS info)
             
            #------------------------- Star/Galaxy Separation ----------------------------
             
            SEEING_FWHM      =3.5,            # stellar FWHM in arcsec
            STARNNW_NAME     ='default.nnw',    # Neural-Network_Weight table filename
             
            #------------------------------ Background -----------------------------------
             
            BACK_TYPE        ='AUTO',           # AUTO or MANUAL
            BACK_VALUE       =0.0,            # Default background value in MANUAL mode
            BACK_SIZE        =64,             # Background mesh: <size> or <width>,<height>
            BACK_FILTERSIZE  =3,              # Background filter: <size> or <width>,<height>
             
            BACKPHOTO_TYPE   ='LOCAL',         # can be GLOBAL or LOCAL
            BACKPHOTO_THICK  =24,             # thickness of the background LOCAL annulus
            BACK_FILTTHRESH  =0.0,            # Threshold above which the background-
                                            # map filter operates
             
            #------------------------------ Check Image ----------------------------------
             
            CHECKIMAGE_TYPE  ='NONE',           # can be NONE, BACKGROUND, BACKGROUND_RMS,
                                            # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                            # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                            # or APERTURES
            CHECKIMAGE_NAME  ='check.fits',     # Filename for the check-image
             
            #--------------------- Memory (change with caution!) -------------------------
             
            MEMORY_OBJSTACK  =3000,           # number of objects in stack
            MEMORY_PIXSTACK  =300000,         # number of pixels in stack
            MEMORY_BUFSIZE   =1024,           # number of lines in buffer
             
            #--------------------------- Experimental Stuff -----------------------------
            
            PSF_NAME         ='default.psf',    # File containing the PSF model
            PSF_NMAX         =1,              # Max.number of PSFs fitted simultaneously
            PATTERN_TYPE     ='RINGS-HARMONIC', # can RINGS-QUADPOLE, RINGS-OCTOPOLE,
                                            # RINGS-HARMONICS or GAUSS-LAGUERRE
            SOM_NAME         ='default.som'    # File containing Self-Organizing Map weights
            )
        
        hdu = fits.open(image)[0]
        hdr = hdu.header
        wcs = WCS(hdr)
        position_pixel = np.array(self.get_radec().to_pixel(wcs = wcs))
        
        conf_param['PHOT_APERTURES'] = float(factor_aperture* hdu.header['SEEING']/self.__obsinfo['pixelscale'])
        conf_param['GAIN'] = float(self.__obsinfo['gain'])
        conf_param['PIXEL_SCALE'] = float(self.__obsinfo['pixelscale'])
        conf_param['BACK_SIZE'] = self.get_targetsize()
        conf_param['SEEING_FWHM'] = hdr['SEEING']
        conf_param['MAG_ZEROPOINT'] = hdr[zpkey]
    
        config = ''
        for param in conf_param.keys():
            config += f'-{param} {conf_param[param]} '
    
        os.chdir('/data2/sextractor')
        os.system(f'source-extractor {image} {config}   ')
        
        seresult = ascii.read('/data2/sextractor/photometry.cat')
        
        result = seresult[(abs(seresult['X_IMAGE']-position_pixel[0])<10/float(self.__obsinfo['pixelscale']))&(abs(seresult['Y_IMAGE']-position_pixel[1])<10/float(self.__obsinfo['pixelscale']))]
        if len(result) >1:
            result = result[np.argmax(result['FLUX_APER'])]
            mag = result['MAG_APER'][0]
            magerr = np.sqrt(hdr['ZPER_4']**2+result['MAGERR_APER'][0]**2)
        elif len(result) ==0:
            return hdr[depthkey], hdr['ZPER_4']
        else:
            mag = result['MAG_APER'][0] 
            magerr = np.sqrt(hdr['ZPER_4']**2+result['MAGERR_APER'][0]**2)
            
        return mag, magerr
    

#%%

