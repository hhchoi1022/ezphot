#%%
from astropy.io import ascii
from astropy.io import fits
from conversion import APASS_to_JH
from astropy.table import Table
from conversion import APASS_format
import os
from conversion import SMSS1_to_JH, SMSS1_to_SDSS
from conversion import PANSTARRS1_to_JH, PANSTARRS1_to_SDSS
from HHsupport_phot import load_sexconfig
import glob
from astropy.stats import sigma_clip
from HHsupport_phot import cross_match, to_skycoord
import numpy as np
import matplotlib.pyplot as plt
import json
from HHsupport_phot import run_sextractor
#%%
class Update_fitsinfo:
    
    def __init__(self, 
                 # basic config
                 imkey = '/data2/*.fits',
                 telescope = None,
                 ccd = None,
                 rasamode = 'High',
                 alltelinfo = '/home/hhchoi1022/Desktop/Gitrepo/config/CCD.dat',
                 alltarinfo = '/home/hhchoi1022/Desktop/Gitrepo/config/alltarget.dat',
                 allcatpath = '/data1/Skycatalog',
                 configdir = '/home/hhchoi1022/Desktop/Gitrepo/config/',
                 savefig =  True,
                 update = True,
                 show = False,
                 # sex config
                 sexdir = '/data2/sextractor',
                 aperfactor = 3,
                 seeing_guess = 5,
                 threshold = 3,
                 # ref config
                 target = None,
                 filter_ = None,
                 refcat = 'PS1',
                 mag_upper = 16,
                 mag_lower = 14,
                 e_mag_upper = 0.1,
                 class_star = 0.9,
                 flag = 1,
                 n_good = 10
                 ):
        # basic config
        self.imlist = glob.glob(imkey)
        self.telescope = telescope
        self.ccd = ccd
        self.rasamode = rasamode
        self.alltelinfo = alltelinfo
        self.alltarinfo = alltarinfo
        self.allcatpath = allcatpath
        self.configdir = configdir
        all_obsinfo = ascii.read(self.alltelinfo, format = 'fixed_width')
        all_obsinfo = all_obsinfo[all_obsinfo['obs'] == self.telescope]
        if len(all_obsinfo) > 1:
            all_obsinfo = all_obsinfo[all_obsinfo['ccd'] == self.ccd]
        if (self.telescope == 'RASA36') & (self.ccd == 'KL4040'):
            if self.rasamode in ['merge','Merge','MERGE']:
                all_obsinfo = all_obsinfo[all_obsinfo['gain'] > 10]
            if self.rasamode in ['high', 'High','HIGH']:
                all_obsinfo = all_obsinfo[all_obsinfo['gain'] < 10]
        self.obsinfo = all_obsinfo
        self.save = savefig
        self.update = update
        self.show = show
        # sex config
        self.sexdir = sexdir
        self.aperfactor = aperfactor
        self.seeing_guess = seeing_guess
        self.threshold = threshold
        # ref config
        self.target = target
        self.filter_ = filter_
        self.refcat = refcat
        self.cutrefstar = (mag_upper, mag_lower, e_mag_upper, class_star, flag, n_good)
    
    def load_config(self, 
                    configfile):
        with open(configfile,'r') as config_json:
            config =  json.load(config_json)
        return config
        
    def make_config(self, filename = 'KCT_updatefits.config'):
            
        config = dict(
                 telescope = 'KCT',
                 ccd = 'STX16803',
                 alltelinfo = '/home/hhchoi1022/Desktop/Gitrepo/config/CCD.dat',
                 alltarinfo = '/home/hhchoi1022/Desktop/Gitrepo/config/alltarget.dat',
                 allcatpath = '/data1/Skycatalog',
                 configdir = '/home/hhchoi1022/Desktop/Gitrepo/config/',
                 savefig =  True,
                 update = True,
                 # sex config
                 sexdir = '/data2/sextractor',
                 aperfactor = 3,
                 seeing_guess = 5,
                 threshold = 3,
                 # ref config
                 target = None,
                 filter_ = None,
                 refcat = 'PS1',
                 mag_upper = 16,
                 mag_lower = 14,
                 e_mag_upper = 0.1,
                 class_star = 0.9,
                 flag = 1,
                 n_good = 10
                )
        json.dump(config, open(self.configdir+filename,'w'), indent = 4, sort_keys = False)
        print(f'Configuration file is made : {self.configdir}{filename}')
        return self.configdir+filename
    
    
    def update_telescope_info(self):
        # For RASA36, multiple modes
        all_telinfo = ascii.read(self.alltelinfo, format = 'fixed_width')
        print(60*"=")
        print(f"Current image : {os.path.basename(self.imlist[0])}")
        print(60*"=")
        print(set(all_telinfo["obs"]))
        self.telescope = input('Select a telescope')
        obs_info = all_telinfo[all_telinfo['obs'] == self.telescope]
        if self.telescope == 'RASA36':
            self.rasamode = input('RASA36 has multiple modes for observation. Please select one. (High/Merge)')
            if self.rasamode in ['merge','Merge','MERGE']:
                obs_info = obs_info[obs_info['gain'] > 10]
            if self.rasamode in ['high', 'High','HIGH']:
                obs_info = obs_info[obs_info['gain'] < 10]
            self.ccd = obs_info['ccd']
            self.obsinfo = obs_info
            return self.obsinfo
        # Other telescopes
        else:
            if len(obs_info) != 1:
                print(set(obs_info["ccd"]))
                self.ccd = input('Multiple CCDs are found. Select one.')
                obs_info = obs_info[obs_info['ccd'] == self.ccd]    
        self.obsinfo = obs_info
        return self.obsinfo
    
    def get_target_info(self, target):
        all_tarinfo = ascii.read(self.alltarinfo, format = 'fixed_width')
        if not target in all_tarinfo['obj']:
            raise AttributeError(f'{target} is not found in target information file')
        else:
            targetinfo = all_tarinfo[all_tarinfo['obj'] == target]
            return targetinfo
    
    def get_image_info(self, image):
        hdr = fits.getheader(image)
        target, filter_, binning = None, None, None
        if 'OBJECT' in hdr.keys():
            target = hdr['OBJECT']
            target.upper()
        if 'FILTER' in hdr.keys():
            filter_ = hdr['FILTER']
        if 'XBINNING' in hdr.keys():
            binning = hdr['XBINNING']
        if ((target != None) & (filter_ != None) & (binning != None)):
            return target, filter_, binning
        else:
            print(hdr)
            while ((target == None) | (filter_ == None)):
                key_target = input('cannot find target information in the header. Please input keywords for target OR break')
                key_filter = input('cannot find filter information in the header. Please input keywords for filter OR break')
                key_binning = input('cannot find binning information in the header. Please input keywords for filter OR break')
                try:
                    target = hdr[key_target]
                    filter_ = hdr[key_filter]
                    binning= hdr[key_binning]
                except:
                    pass
                if (key_target == 'break') | (key_filter == 'break') | (key_binning== 'break'):
                    raise AttributeError("target/filter/binning cannot be found in the header")
            target.upper()
            return target, filter_, binning
        
    def load_APASS(self, target, filter_,
                   mag_upper = 18,
                   mag_lower = 14,
                   e_mag_upper = 0.1,
                   class_star = None,
                   flag = None,
                   n_good = None):
        
        APASS_file = f'{self.allcatpath}/APASS/{target}.csv'
        if not os.path.isfile(APASS_file):
            result_tbl = None
            refcat = None
            print(f'{target} not exist in APASS catalog')
            return result_tbl, refcat
        if filter_ in ['B','V','R','I']:
            APASS_table = APASS_to_JH(APASS_file)
            refcat = 'APASS'
        elif filter_ in ['g','r','i']:
            APASS_table = APASS_format(APASS_file)
            refcat = 'APASS'
        else:
            result_tbl = None
            refcat = None
            print(f'{filter_} filter not exist in APASS catalog')
            return result_tbl, refcat
        cutline = (
            (APASS_table[f'{filter_}_mag'] < mag_upper)&
            (APASS_table[f'{filter_}_mag'] > mag_lower)&
            (APASS_table[f'e_{filter_}_mag'] < e_mag_upper)
            )
        result_tbl = APASS_table[cutline]
        return result_tbl, refcat
            
    def load_SMSS(self, target, filter_,
                  mag_upper = 18,
                  mag_lower = 14,
                  e_mag_upper = 0.1,
                  class_star = None,
                  flag = None,
                  n_good = None):
        
        SMSS_file = f'{self.allcatpath}/Skymapper/{target}.csv'
        if not os.path.isfile(SMSS_file):
            result_tbl = None
            refcat = None
            print(f'{target} not exist in Skymapper catalog')
            return result_tbl, refcat
        if filter_ in ['B','V','R','I']:
            SMSS_table = SMSS1_to_JH(SMSS_file)
            refcat = 'Skymapper'
        elif filter_ in ['g','r','i']:
            SMSS_table = SMSS1_to_SDSS(SMSS_file)
            refcat = 'Skymapper'
        else:
            result_tbl = None
            refcat = None
            print(f'{filter_} filter not exist in Skymapper catalog')
            return result_tbl, refcat
        cutline = (
            (SMSS_table[f'{filter_}_mag'] < mag_upper)&
            (SMSS_table[f'{filter_}_mag'] > mag_lower)&
            (SMSS_table[f'e_{filter_}_mag'] < e_mag_upper)&
            (SMSS_table['class_star'] > class_star)&
            (SMSS_table['flag'] <= flag)&
            (SMSS_table['ngood'] > n_good)
            )
        result_tbl = SMSS_table[cutline]
        return result_tbl, refcat
    
    def load_PS1(self, target, filter_,
                 mag_upper = 18,
                 mag_lower = 14,
                 e_mag_upper = 0.1,
                 class_star = 0.9,
                 flag = None,
                 n_good = None):
        
        PS_file = f'{self.allcatpath}/PanSTARRS1/{target}.csv'
        if not os.path.isfile(PS_file):
            result_tbl = None
            refcat = None
            print(f'{target} not exist in PanSTARRS catalog')
            return result_tbl, refcat
        if filter_ in ['B','V','R','I']:
            PS_table = PANSTARRS1_to_JH(PS_file)
            refcat = 'PanSTARRS'
        elif filter_ in ['g','r','i']:
            PS_table = PANSTARRS1_to_SDSS(PS_file)
            refcat = 'PanSTARRS'
        else:
            result_tbl = None
            refcat = None
            print(f'{filter_} filter not exist in PanSTARRS catalog')
            return result_tbl, refcat
        cutline = (
            (PS_table[f'{filter_}_mag'] < mag_upper)&
            (PS_table[f'{filter_}_mag'] > mag_lower)&
            (PS_table[f'e_{filter_}_mag'] < e_mag_upper)&
            (PS_table[f'{filter_}_mag']-PS_table[f'{filter_}_Kmag'] < 1-class_star)
            )
        result_tbl = PS_table[cutline]
        return result_tbl, refcat
    
    def update_hdr(self, image, hdrkey, hdrvalues, hdrcomments):
        for key, value, comment in zip(hdrkey, hdrvalues, hdrcomments):
            fits.setval(image, key, value = value, comment = comment)
    
    def run(self):
        while len(self.obsinfo) != 1:
            self.update_telescope_info()
        conf_param = load_sexconfig()
        conf_param['CATALOG_NAME'] = 'zeropoint.cat'
        conf_param['PARAMETERS_NAME'] = 'zeropoint.SEparam'
        conf_param['DETECT_THRESH'] = self.threshold
        conf_param['SEEING_FWHM'] = self.seeing_guess
        conf_param['GAIN'] = float(self.obsinfo['gain'])
        conf_param['SATUR_LEVEL'] = 65536
        
        for image in self.imlist:
            
            # Load basic information for selected telescope and target
            target, filter_, binning = self.get_image_info(image)
            if binning != None:
                bin_factor = binning / self.obsinfo['binning']
                
            else:
                bin_factor = 1
            self.obsinfo['pixelscale'] = self.obsinfo['pixelscale'] * bin_factor
            pixsize = round(float(self.obsinfo['pixelscale']),3)
            targetinfo = self.get_target_info(target)
            bkgsize = targetinfo['maxaxis'][0]*60/pixsize
            conf_param['PIXEL_SCALE'] = float(self.obsinfo['pixelscale'])
            conf_param['BACK_SIZE'] = int(bkgsize)
            
            # Load sky reference catalog
            if self.refcat.upper() == 'APASS':
                sky_tbl, refkey = self.load_APASS(target, filter_, *self.cutrefstar)
            if self.refcat.upper() == 'PS1':
                sky_tbl, refkey = self.load_PS1(target, filter_, *self.cutrefstar)
            if self.refcat.upper() == 'SKYMAPPER':
                sky_tbl, refkey = self.load_SMSS(target, filter_, *self.cutrefstar)
            if not self.refcat.upper() in ['APASS','PS1','SKYMAPPER']:
                sky_tbl, refkey = self.load_APASS(target, filter_, *self.cutrefstar)
                if refkey == None:
                    sky_tbl, refkey = self.load_PS1(target, filter_, *self.cutrefstar)
                if refkey == None:
                    sky_tbl, refkey = self.load_SMSS(target, filter_, *self.cutrefstar)
                if refkey == None:
                    print(f'No reference catalog exists for {target}')
                    break
            
            # Run source extractor 
            mag_upper, mag_lower, e_mag_upper, class_star, flag, n_good = self.cutrefstar
            obj_tbl1 = run_sextractor(image, conf_param)
            obj_tbl1_cut = obj_tbl1[
                                (obj_tbl1['FLAGS'] <= flag)&
                                (obj_tbl1['MAGERR_AUTO'] < e_mag_upper)&
                                (3600*obj_tbl1['FWHM_WORLD'] > 1)#&
                                #(3600*obj_tbl1['FWHM_WORLD'] < 5) 
                               ]
            
            sky_coords = to_skycoord(sky_tbl['ra'], sky_tbl['dec'])
            obj_coords1 = to_skycoord(obj_tbl1_cut['ALPHA_J2000'], obj_tbl1_cut['DELTA_J2000'])
            
            matched_obj_idx1, matched_sky_idx1, _ = cross_match(obj_coords1, sky_coords, self.seeing_guess/pixsize)
            
            if len(matched_obj_idx1) < 2:
                raise AttributeError(f'Matching failed. The number of matched stars is {len(matched_obj_idx1)}')
            
            seeing1 = round(3600*np.median(sigma_clip(obj_tbl1_cut[matched_obj_idx1]['FWHM_WORLD'],sigma=3,maxiters=1).data),3)
            conf_param['PHOT_APERTURES'] = round(self.aperfactor*seeing1/pixsize, 3)
            conf_param['SEEING_FWHM'] = seeing1
            
            obj_tbl2 = run_sextractor(image, conf_param)
            obj_tbl2_cut = obj_tbl2[
                                (obj_tbl2['FLAGS'] <= flag)&
                                (obj_tbl2['MAGERR_AUTO'] < e_mag_upper)&
                                (3600*obj_tbl2['FWHM_WORLD'] > 1)&
                                #(3600*obj_tbl2['FWHM_WORLD'] < 5)&
                                (obj_tbl2['CLASS_STAR'] > class_star)
                               ]
            obj_coords2 = to_skycoord(obj_tbl2_cut['ALPHA_J2000'], obj_tbl2_cut['DELTA_J2000'])
            matched_obj_idx2, matched_sky_idx2, _ = cross_match(obj_coords2, sky_coords, 1.5*seeing1)
            
            #### Edit
            matched_obj = obj_tbl2_cut[matched_obj_idx2]
            matched_sky = sky_tbl[matched_sky_idx2]
            matched_delmag = matched_sky[f'{filter_}_mag'] - matched_obj['MAG_APER'] 
            zpclip = sigma_clip(matched_delmag, sigma = 3, maxiters = 1)
            refstar_obj = matched_obj[~zpclip.mask]
            refstar_sky = matched_sky[~zpclip.mask]
            seeing2 = round(np.median(3600*refstar_obj['FWHM_WORLD']),3)
            conf_param['PHOT_APERTURES'] = round(self.aperfactor*seeing2/pixsize, 3)
            conf_param['SEEING_FWHM'] = seeing2
            obj_tbl3 = run_sextractor(image, conf_param)
            obj_tbl3_cut = obj_tbl3[
                    (obj_tbl3['FLAGS'] <= flag)&
                    (obj_tbl3['MAGERR_AUTO'] < e_mag_upper)&
                    (3600*obj_tbl3['FWHM_WORLD'] > 1)&
                    #(3600*obj_tbl2['FWHM_WORLD'] < 5)&
                    (obj_tbl3['CLASS_STAR'] > class_star)
                    ]
            obj_coords3 = to_skycoord(obj_tbl3_cut['ALPHA_J2000'], obj_tbl3_cut['DELTA_J2000'])
            matched_obj_idx3, matched_sky_idx3, _ = cross_match(obj_coords3, sky_coords, 1.5*seeing2)
            matched_obj = obj_tbl3_cut[matched_obj_idx3]
            matched_sky = sky_tbl[matched_sky_idx3]
            ####
            
            
            # Selesct reference star
            #matched_obj = obj_tbl2_cut[matched_obj_idx2]
            #matched_sky = sky_tbl[matched_sky_idx2]
            matched_delmag = matched_sky[f'{filter_}_mag'] - matched_obj['MAG_APER'] 
            zpclip = sigma_clip(matched_delmag, sigma = 3, maxiters = 1)
            refstar_obj = matched_obj[~zpclip.mask]
            refstar_sky = matched_sky[~zpclip.mask]
            
            # Calculate all parameters 
            refstar_delmag = refstar_sky[f'{filter_}_mag'] - refstar_obj['MAG_APER']
            zp = round(np.median(refstar_delmag),3)
            seeing = round(np.median(3600*refstar_obj['FWHM_WORLD']),3)
            zperr = round(np.std(refstar_delmag),3)
            skysig = round(obj_tbl2_cut['THRESHOLD'][0]/self.threshold,3)
            depth_5sig = round(-2.5*np.log10(5*skysig*np.sqrt(np.pi*((self.aperfactor/2*seeing/pixsize)**2))) + zp,3)
            depth_3sig = round(-2.5*np.log10(3*skysig*np.sqrt(np.pi*((self.aperfactor/2*seeing/pixsize)**2))) + zp,3)
            
            # Update header
            if self.update:
                hdrkeys = ['HH_ZP','HH_SEEING','HH_e_ZP','HH_SKYSIG', 'HH_DEPTH5', 'HH_DEPTH3', 'HH_REFCAT']
                hdrvalues = [zp, seeing, zperr, skysig, depth_5sig, depth_3sig, refkey]
                hdrcomments = [f'APERSIZE = {self.aperfactor}*SEEING', '', '', '', f'APERSIZE = {self.aperfactor}*SEEING', f'APERSIZE = {self.aperfactor}*SEEING', '']
                self.update_hdr(image, hdrkeys, hdrvalues, hdrcomments) 
         
                plt.figure(dpi = 100, figsize =(10,5))
                plt.title(image)
                plt.scatter(matched_sky[f'{filter_}_mag'], matched_delmag, facecolor = 'none', edgecolor = 'r', alpha = 0.3, label = 'clipped stars')
                plt.errorbar(matched_sky[f'{filter_}_mag'], matched_delmag, np.sqrt(matched_sky[f'e_{filter_}_mag']**2+matched_obj['MAGERR_APER']**2), fmt = 'none', elinewidth = 1, c = 'r', capsize = 3, alpha = 0.3)
                plt.scatter(refstar_sky[f'{filter_}_mag'], refstar_delmag, facecolor = 'none', edgecolor = 'k', label = 'reference stars')
                plt.errorbar(refstar_sky[f'{filter_}_mag'], refstar_delmag, np.sqrt(refstar_sky[f'e_{filter_}_mag']**2+refstar_obj['MAGERR_APER']**2), fmt = 'none', elinewidth = 1, c = 'k', capsize = 3)
                plt.xlabel('Mag[AB]')
                plt.ylabel('ZP[AB]')
                plt.xlim(np.min(refstar_sky[f'{filter_}_mag'])-1, np.max(refstar_sky[f'{filter_}_mag']+1))
                plt.ylim(np.min(refstar_delmag)-1, np.max(refstar_delmag)+1)
                plt.axhline(zp, c = 'k', linewidth = 1, linestyle = '--', label = r'ZP = %.3f$\pm$%.3f'%(zp,zperr))
                plt.fill_between([np.min(refstar_sky[f'{filter_}_mag'])-1, np.max(refstar_sky[f'{filter_}_mag']+1)],[zp-zperr],[zp+zperr], alpha = 0.3)
                plt.text(np.min(refstar_sky[f'{filter_}_mag'])-0.9,np.max(refstar_delmag)+0.3,f'Depth[5sig] : {depth_5sig}\nSeeing : {seeing}\nRefcat : {refkey}')
                plt.legend(loc = 1)
            if self.save:
                plt.savefig(f'{image.replace(".fits","_zpcalc")}.png')
            if self.show:            
                plt.show()  


            
# %%
