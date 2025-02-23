#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:33:40 2022

@author: hhchoi1022
"""

# %%

# Photometry for a designated point source 

# Creating Aperture Objects from RADec
from astropy.io import ascii
import os
from astropy.io import fits
import glob
from HHsupport_phot import get_imginfo
import json
from HHsupport_analysis import groupping_table
from HHsupport_phot import align_img, combine_img, cutout_img, subtract_img
from HHsupport_phot import get_obsinfo
from update_fitsinfo import Update_fitsinfo
from astropy.wcs import FITSFixedWarning
import warnings
from time import sleep
from astropy.table import Table
from HHsupport_phot import to_skycoord
from astropy.table import vstack
from HHsupport_phot import sleep
from HHsupport_phot import select_sources
from HHsupport_phot import load_sexconfig
from HHsupport_phot import get_tarinfo
from HHsupport_phot import run_sextractor
from HHsupport_phot import cross_match
from HHsupport_phot import printProgress
import numpy as np


#%%



class Photometry:
    
    def __init__(self, 
                 imkey = None,
                 configfile = None,
                 ):
        warnings.simplefilter('ignore', category=FITSFixedWarning)
        warnings.simplefilter('ignore', category=UserWarning)
        self.configdir = '/home/hhchoi1022/Desktop/Gitrepo/config/'
        if configfile == None:
            print('Configuration file not found. Run Photometry.make_config() and insert configuration file path')
        else:
            self.config = self.load_config(configfile)
            if imkey == None:
                imkey = self.config['imkey']
            print('============== LOADING IMAGE INFORMATION...')
            imlist = glob.glob(imkey)
            self.imginfo = get_imginfo(imlist)
            self.obsinfo = get_obsinfo(self.config['observatory'],self.config['ccd'],self.config['rasamode'])
            self.updateconfig = self.load_config(self.config['update']['updateconfigfile'])
            self.updateconfig['aperfactor'] = self.config['photometry']['aperfactor']
            print(f'============== {len(self.imginfo)} images are loaded!\n')
            print('============== Start processing... =============')
            print('============== MODE')
            print(f'============== combine    : {self.config["combine"]["mode"]}')
            print(f'============== update     : {self.config["update"]["mode"]}')
            print(f'============== subtract   : {self.config["subtract"]["mode"]}')
            print(f'============== photometry : {self.config["photometry"]["mode"]}')
            print(f'============== remove     : {self.config["remove"]["mode"]}')
            print('=================================================')
        
    def select_outlier(self, 
                        all_tbl,
                        sigma = 3,
                        remove = False, 
                        show = True):
        from astropy.stats import sigma_clip
        from astropy.table import Table
        from astropy.io import fits
        import matplotlib.pyplot as plt
        cut_tbl = all_tbl[(sigma_clip(all_tbl[self.config["photometry"]["depth_key"]],sigma = sigma).mask) | (sigma_clip(all_tbl[self.config["photometry"]["seeing_key"]],sigma = sigma).mask)]
        selected_tbl = all_tbl[~(sigma_clip(all_tbl[self.config["photometry"]["depth_key"]],sigma = sigma).mask) & (~sigma_clip(all_tbl[self.config["photometry"]["seeing_key"]],sigma = sigma).mask)]
        
        if show == True:
            plt.figure(figsize = (10,5))
            plt.scatter(all_tbl[self.config["photometry"]["seeing_key"]],all_tbl[self.config["photometry"]["depth_key"]], c= 'k', marker = 'o', alpha = 0.6)
            plt.scatter(cut_tbl[self.config["photometry"]["seeing_key"]],cut_tbl[self.config["photometry"]["depth_key"]], c= 'r', label = 'Outlier', marker = 'o', alpha = 0.6)
            plt.xlabel('Seeing[arcsec]')
            plt.ylabel('Depth[AB]')
            plt.grid()
            plt.legend()
            plt.show()
        if (remove == True) & (len(cut_tbl) > 0):
            for image in cut_tbl['file']:
                os.system(f'rm {image}')
        return selected_tbl, cut_tbl
        
    def load_config(self, 
                    configfile):
        with open(configfile,'r') as config_json:
            config =  json.load(config_json)
        return config

    def make_config(self, filename = 'photometry.config'):
        observatory = 'KCT'
        ccd = 'STX16803'
        combine = True
        combine_range = 0.1 # days       
        updateconfigfile = '/home/hhchoi1022/Desktop/Gitrepo/config/KCT_STX16803_updatefits.config'
        remove = True
        update = True
        subtract = True
        refdir = '/data1/Reference/KCT_STX16803/'
        refim = None
        rasamode = 'High'
        sources = None
        photometry = True
        aperfactor = 3
        seeing_key = 'hh_seeing'
        zp_key = 'hh_zp'
        outfile = '../photometry/resultphot.fixascii'
        match_distance_second = 3
        depth_key = 'hh_depth5'
        zperr_key = 'hh_e_zp'
        cutout = True
        cutout_size = 2000
        phot_threshold = 3
        imkey = '/data2/SN2021aefx_220627/KCT_STX16803/?/Calib*.fits'
        config = {
                # observatory info
                "imkey"      : imkey,
                'observatory': observatory,
                "ccd"        : ccd,
                "rasamode"   : rasamode,
                
                # method info
                "combine"    :{"mode": combine,
                               "range": combine_range},
                "update"     :{"mode": update,
                               "updateconfigfile": updateconfigfile,
                               },
                "cutout"     :{"mode": cutout,
                               "size": cutout_size},
                "subtract"   :{"mode": subtract,
                               "refdir": refdir,
                               "refim": refim},
                "photometry" :{"mode": photometry,
                               "sources": sources,
                               "aperfactor": aperfactor,
                               "threshold" : phot_threshold,
                               "seeing_key": seeing_key,
                               "zp_key": zp_key,
                               "zperr_key": zperr_key,
                               "depth_key": depth_key,
                               "match_distance_second" : match_distance_second},
                "remove"     :{"mode": remove},
                # set directory
                "outfile"    : outfile
                }
        json.dump(config, open(self.configdir+filename,'w'), indent = 4, sort_keys = False)
        print(f'Configuration file is made : {self.configdir}{filename}')
        return self.configdir+filename

    
    def run(self):
        ##### Image process and selection 
        filter_groups = self.imginfo.group_by('filter').groups
        phot_groups = Table()
        updated_keywords = [self.config['photometry']['zp_key'],self.config['photometry']['zperr_key'],self.config['photometry']['seeing_key'],self.config['photometry']['depth_key']]
        for filter_group in filter_groups:
            filter_ = filter_group['filter'][0]
            
            # IF update == True, Update image
            if self.config['update']['mode']:
                updatedlist = []
                for i, image in enumerate(filter_group['file']):
                    printProgress(i, len(filter_group), prefix = f'Updating {filter_} band[{i}/{len(filter_group)}]')
                    try:
                        Update_fitsinfo(imkey = image, **self.updateconfig).run()
                        updatedlist.append(image)
                    except:
                        print('Image information update failed')
                        pass
                
                filter_group = get_imginfo(updatedlist, updated_keywords)
            #else:
            #    filter_group = get_imginfo(filter_group['file'], updated_keywords)

            # IF combine == True, Combine image
            if self.config['combine']['mode']:
                #filter_group, _ = self.select_outlier(filter_group, sigma = 3, remove = False)
                filter_group = groupping_table(filter_group,'jd',tolerance = self.config['combine']['range'])
                combine_groups = filter_group.group_by('group').groups
                combinedlist = []
                for i, combine_group in enumerate(combine_groups):
                    printProgress(i, len(combine_groups), prefix = f'Combining {filter_} band[{i}/{len(combine_groups)}]')
                    ncombine = len(combine_group)
                    refimage = combine_group[ncombine//2]['file']
                    alignedlist = []
                    for image in combine_group['file']:
                        try:
                            aligned_path = align_img(image, refimage, prefix = 'align_')
                            alignedlist.append(aligned_path)
                        except:
                            pass
                    try:
                        combined_path = combine_img(alignedlist, clip = 'extrema', clip_extrema_nlow = 0, clip_extrema_nhigh =1, scale = 'zero', refim = refimage)
                        Update_fitsinfo(imkey = combined_path, **self.updateconfig).run()
                        combinedlist.append(combined_path)  
                    except:
                        pass                  
                    if self.config['remove']['mode']:
                        for file in alignedlist:
                            os.system(f'rm {file}')
                filter_group = get_imginfo(combinedlist, updated_keywords)

            # IF subtract == True, Subtract image
            if self.config['subtract']['mode']:
                if self.config['subtract']['refim'] == None:
                    target = filter_group['object'][0]
                    filter_ = filter_group['filter'][0]
                    refimages = os.listdir(self.config['subtract']['refdir'])
                    ref_candidates = [self.config['subtract']['refdir']+image for image in refimages if (target in image) & (image.endswith('.fits'))]
                    # For RASA36, 2 modes
                    if (self.config['observatory'] == 'RASA36') & (self.config['ccd'] == 'KL4040'):
                        refimage = [ref_candidate for ref_candidate in ref_candidates if self.config['rasamode'].lower() in ref_candidate.lower()][0]
                    else:
                        refimage = [ref_candidate for ref_candidate in ref_candidates if fits.getheader(ref_candidate)['filter'] == filter_][0]
                else:
                    refimage = self.config['subtract']['refim']
                if self.config['cutout']['mode']:
                    refimage_cut = cutout_img(refimage, size = self.config['cutout']['size'])
                subtractlist = []
                cutoutlist = []
                alignedlist = []
                for i, image in enumerate(filter_group['file']):
                    printProgress(i, len(filter_group), prefix = f'Subtracting {filter_} band[{i}/{len(filter_group)}]')
                    try:
                        aligned_path = align_img(image, refimage)
                        alignedlist.append(aligned_path)
                        if self.config['cutout']['mode']:
                            cutout_path = cutout_img(aligned_path, size = self.config['cutout']['size'])
                            cutoutlist.append(cutout_path)
                            subtract_path = subtract_img(cutout_path, refimage_cut)
                            
                        else:
                            subtract_path = subtract_img(aligned_path, refimage)
                        subtractlist.append(subtract_path)
                    except:
                        pass
                if self.config['cutout']['mode']:
                    os.system(f'rm -rf {refimage_cut}')
                if self.config['remove']['mode']:
                    for file in cutoutlist:
                        os.system(f'rm -rf {file}')
                    for file in alignedlist:
                        os.system(f'rm -rf {file}')
                filter_group = get_imginfo(subtractlist, updated_keywords)
            phot_groups = vstack([filter_group, phot_groups])
        self.photinfo = get_imginfo(phot_groups['file'], updated_keywords)

        ##### Image process and selection END ##########
        
        ##### Source selection
        try:
            self.sources = ascii.read(self.config['photometry']['sources'], format = 'fixed_width')
        except:
            mode = 0
            while mode != '4':
                print('No catalog file found for photometry. Select sources for photometry')
                print('Select the mode to select a source[s]')
                print('1. Select sources with the coordinates of the target (RA, Dec)')
                print('2. Select sources in the visulized image (IRAF is needed)')
                print('3. Halt the process')
                mode = input('MODE(1,2,3)...?')
                if mode == '1':
                    ra = input('Input RA of the target(single)')
                    dec = input('Input Dec of the target(single)')
                    coords = to_skycoord(ra,dec)
                    self.sources = Table(data = [[1], [coords.ra.value], [coords.dec.value], ['']], names = ['ID','ra','dec','note'])
                    self.sources['ID'] = 1
                    self.sources['ra'] = coords.ra.value
                    self.sources['dec'] = coords.dec.value
                    self.sources['note'] = ''
                    self.sources.write(f'{os.path.dirname(self.photinfo["file"][0])}/sources.cat', format = 'ascii.fixed_width')
                    mode = '4'
                if mode == '2':
                    #try:
                    sourcefile = select_sources(self.photinfo[np.random.randint(1,len(self.photinfo))]['file'])
                    if not sourcefile ==None:
                        self.sources = ascii.read(sourcefile, format = 'fixed_width')
                        mode = '4'
                #except:     
                    #    pass
                if mode == '3':
                    mode = '4'
                    raise AttributeError('Abort!')
        ##### Source selection END ##########

        ##### Photometry (images : self.photinfo, sources : self.sources)
        if self.config['photometry']['mode']:
            result_all = Table()
            for i, imageinfo in enumerate(self.photinfo):
                printProgress(i, len(self.photinfo), prefix = f'Photometry [{i}/{len(self.photinfo)}]')
                conf_param = load_sexconfig()
                
                binning = imageinfo['xbinning']
                if binning != None:
                    bin_factor = binning / self.obsinfo['binning']
                self.obsinfo['pixelscale'] = round(float(self.obsinfo['pixelscale'] * bin_factor),3)
                pixsize = round(float(self.obsinfo['pixelscale']),3)
                
                #targetinfo = get_tarinfo(imageinfo['object'])
                #bkgsize = targetinfo['maxaxis'][0]*60/pixsize
                
                jd = imageinfo['jd']
                image = imageinfo['file']
                filter_ = imageinfo['filter']
                depth = imageinfo[self.config['photometry']['depth_key']]
                seeing = imageinfo[self.config['photometry']['seeing_key']]
                zp = imageinfo[self.config['photometry']['zp_key']]
                zperr = imageinfo[self.config['photometry']['zperr_key']]
                
                conf_param['PIXEL_SCALE'] = float(self.obsinfo['pixelscale'])
                #conf_param['BACK_SIZE'] = int(bkgsize)
                conf_param['PHOT_APERTURES'] = float(self.config['photometry']['aperfactor']* seeing/self.obsinfo['pixelscale'])
                conf_param['GAIN'] = float(self.obsinfo['gain'])
                conf_param['SEEING_FWHM'] = seeing            
                conf_param['MAG_ZEROPOINT'] = zp
                conf_param['CATALOG_NAME'] = 'photometry.cat'
                conf_param['PARAMETERS_NAME'] = 'photometry.SEparam'
                conf_param['BACKPHOTO_TYPE'] = 'LOCAL'
                conf_param['BACKPHOTO_THICK'] = float(self.config['photometry']['aperfactor']* seeing/self.obsinfo['pixelscale'])
                conf_param['DETECT_THRESH'] = self.config['photometry']['threshold']

                sex_result = run_sextractor(image, conf_param)
                sex_catalog = to_skycoord(sex_result['ALPHA_J2000'],sex_result['DELTA_J2000'])
                source_catalog = to_skycoord(self.sources['ra'], self.sources['dec'])
                cat_idx, source_idx, _ = cross_match(sex_catalog, source_catalog, max_distance_second= self.config['photometry']['match_distance_second'])
                '''
                if len(source_idx) == 0:
                    result_catalog = self.sources[[0]]
                    result_single = sex_result[[0]]
                    result_single['ID'] = result_catalog['ID']
                    result_single['filter'] = filter_
                    result_single['seeing'] = seeing
                    result_single['jd'] = jd
                    result_single['file'] = image
                    result_single['zp'] = zp
                    result_single['zperr'] = zperr
                    result_single['depth'] = depth
                    result_single['NUMBER'] = 0
                    result_single['FLUX_APER'] = 0
                    result_single['FLUXERR_APER'] = 0
                    result_single['MAG_APER'] = 0
                    result_single['MAGERR_APER'] = 0
                    result_all = vstack([result_single, result_all])
                '''
                #else:
                    #cat_idx = [cat_idx[np.argmax(sex_result[cat_idx]['FLUX_APER'])]]
                if not len(source_idx) == 0:
                    result_catalog = self.sources[source_idx]
                    result_single = sex_result[cat_idx].copy()
                    result_single['ID'] = result_catalog['ID']
                    result_single['filter'] = filter_
                    result_single['seeing'] = seeing
                    result_single['jd'] = jd
                    result_single['file'] = image
                    result_single['zp'] = zp
                    result_single['zperr'] = zperr
                    result_single['depth'] = depth
                    result_single = result_single[np.argmin(result_single['MAG_APER'])]
                    result_all = vstack([result_single, result_all])
            if 'filter' in result_all.keys():
                result_all.sort(['filter','jd'])
                result_all['observatory'] = self.config['observatory']
                result_all.rename_columns(['MAG_APER','MAGERR_APER','jd','zp','zperr','ALPHA_J2000','DELTA_J2000'],['mag','e_mag','obsdate','zp','e_zp','ra','dec'])
                result_all.remove_columns(['NUMBER','FLUX_APER','FLUXERR_APER','X_IMAGE','Y_IMAGE','X_WORLD','Y_WORLD'])
                result_all.write(f'{self.config["outfile"]}', format = 'ascii.fixed_width', overwrite = True)
                print(f'Output file : {self.config["outfile"]}')
        ##### Photometry END    
        
            
        
        
#%%
phot = Photometry(configfile = '/home/hhchoi1022/Desktop/Gitrepo/config/LSGT_STX16803_photometry.config')
phot.run()   
#%%
phot = Photometry(configfile = '/home/hhchoi1022/Desktop/Gitrepo/config/KCT_STX16803_photometry.config')
#%%
phot.run()   
        
        

#%%
import sys
if __name__ == '__main__':
    if len(sys.argv)>2:
        file_path = sys.argv[1]
        imkey = sys.argv[2]
        phot = Photometry(imkey = imkey, configfile = file_path)
        phot.run()
    else:
        file_path = sys.argv[1]	
        phot = Photometry(configfile = file_path)
        phot.run()

           

#%%
#phot = Photometry(imkey = '/data2/SN2021aefx_220627/LSGT/?/Calib*.fits', configfile = '../config/LSGT_STX16803_photometry.config')

phot = Photometry(configfile ='../config/KCT_GRB220921A.config')


# %%
