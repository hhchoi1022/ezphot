#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 22:05:35 2022

@author: hhchoi1022
"""

''
import os, glob
os.chdir('/home/hhchoi1022/Desktop/Gitrepo/photometry')
from photometry import Photometry
from astropy.io import ascii
from image_evaluation import UPDATE_IMAGE
import warnings
warnings.filterwarnings('ignore')
import sys


def phot(imkey, refim, obs_info, ra, dec, program, write = False, sub = True, start = True):
    import re
    from astropy.io import fits
    from astropy.time import Time
    import matplotlib.pyplot as plt
    from astropy.table import Table
    path = os.path.dirname(imkey)
    phot = Photometry(imkey = imkey, refimage = refim, obsinfo = obs_info, obj_ra = ra, obj_dec= dec)
    
    #if obs_info['obs'] == 'RASA36':
    #    phot.divide_RASA()
        
    selected, outlying = phot.select_outlier(remove = False, check = True) 
    
    print(50*'=')
    print(f'{len(outlying)} outlying images are removed.')
    print(f'Start Photonmetry of the target ({len(selected)} images)')
    print(50*'=')
    print('Image aligning...')

    aligned_hdulist = []
    aligned_imlist = []
    if start:
        for i, image in enumerate(selected['Image']):
            try:
                if write == True:
                    aligned_hdu, aligned_image = phot.align(image, scale = False, write = True)
                    aligned_imlist.append(aligned_image)
                    imdir = os.path.dirname(aligned_image)
                else:
                    aligned_hdu, aligned_image = phot.align(image, scale = False, write = False)
                    aligned_hdulist.append(aligned_hdu)
                    aligned_imlist.append(aligned_image)
                    imdir = os.path.dirname(image)
            except:
                print(f'{image} cannot be aligned')
            printProgress(i, len(selected), f'Progress[{i}]:', f'Complete[{len(selected)}]', 1, 50)
    else:
        aligned_imlist = sorted(glob.glob(path+'/aligned/*.fits'))
        imdir = path+'/aligned'
        if len(aligned_imlist) == 0:
            for i, image in enumerate(selected['Image']):
                try:
                    if write == True:
                        aligned_hdu, aligned_image = phot.align(image, scale = False, write = True)
                        aligned_imlist.append(aligned_image)
                        imdir = os.path.dirname(aligned_image)
                    else:
                        aligned_hdu, aligned_image = phot.align(image, scale = False, write = False)
                        aligned_hdulist.append(aligned_hdu)
                        aligned_imlist.append(aligned_image)
                        imdir = os.path.dirname(image)
                except:
                    print(f'{image} cannot be aligned')
                printProgress(i, len(selected), f'Progress[{i}]:', f'Complete[{len(selected)}]', 1, 50)

    os.chdir(f'{imdir}')
    os.makedirs(f'{imdir}/combined', exist_ok = True)
    os.chdir(f'{imdir}/combined')
    print(50*'=')
    print(f'Image combining...')
    datelist = []
    for image in aligned_imlist:
        datelist.append(re.findall('(20\d\d\d\d\d\d)', os.path.basename(image))[0])
    print(f'Total {len(set(datelist))} days for photometry')
    print(f'Image combining...')
    if start:
        for i, date in enumerate(set(datelist)):
            if write == False:
                comlist = [aligned_hdulist[i] for i in range(len(aligned_hdulist)) if date in aligned_imlist[i]]
                phot.combine_ccdproc(comlist, write = True)
                printProgress(i, len(set(datelist)), f'Progress[{i}]:', f'Complete[{len(set(datelist))}]', 1, 50)
            else:
                comlist = [aligned_imlist[i] for i in range(len(aligned_imlist)) if date in aligned_imlist[i]]
                phot.combine_ccdproc(comlist, write = True)
                printProgress(i, len(set(datelist)), f'Progress[{i}]:', f'Complete[{len(set(datelist))}]', 1, 50)
    else:
        combined_imlist = sorted(glob.glob(f'{imdir}/combined/*com.fits'))
        if len(combined_imlist) == 0:
            for i, date in enumerate(set(datelist)):
                if write == False:
                    comlist = [aligned_hdulist[i] for i in range(len(aligned_hdulist)) if date in aligned_imlist[i]]
                    phot.combine_ccdproc(comlist, write = True)
                    printProgress(i, len(set(datelist)), f'Progress[{i}]:', f'Complete[{len(set(datelist))}]', 1, 50)
                else:
                    comlist = [aligned_imlist[i] for i in range(len(aligned_imlist)) if date in aligned_imlist[i]]
                    phot.combine_ccdproc(comlist, write = True)
                    printProgress(i, len(set(datelist)), f'Progress[{i}]:', f'Complete[{len(set(datelist))}]', 1, 50)
                
    aligned_hdulist = []
    combined_imlist = sorted(glob.glob(f'{imdir}/combined/*com.fits'))
    cutout_imlist = []
    print(50*'=')
    print('Image zeropoint calculation...')
    for image in combined_imlist:
        UPDATE_IMAGE(image, phot.get_obsinfo(), refcat = 'APASS', check = True)
        _, cutout_image = phot.cutout(image, write = True, cutoutsize = 1000)
        cutout_imlist.append(cutout_image)
    
    UPDATE_IMAGE(refim, phot.get_obsinfo(), refcat='APASS', check = True)
    _, refimage = phot.cutout(refim, write = True, cutoutsize = 1000)
    
    cutout_imlist = sorted(glob.glob(f'{imdir}/combined/cutout/*com.fits'))
    subtracted_imlist = cutout_imlist
    if sub == True:
        print(50*'=')
        print('Image subtraction...')
        for image in cutout_imlist:
            phot.subtraction_hotpants(image, refimage)
        imdir = os.path.dirname(image)
        subtracted_imlist = sorted(glob.glob(f'{imdir}/sub*.fits'))#f'{imdir}/aligned/combined/cutout/hd*.fits')) 
    print(50*'=')
    print('Image Photometry...')
    maglist = []
    magerrlist = []
    obsdatelist = []
    limmaglist = []
    limmagerrlist = []
    fir_image = phot.get_imlist()[0]
    detectiondate = round(Time(fits.getheader(fir_image)['JD'], format = 'jd').mjd,5)
    for image in subtracted_imlist:
        hdu = fits.open(image)[0]
        limitmag = hdu.header['UL5_4']
        limitmagerr = hdu.header['ZPER_4']
        if program == 'photutils':
            mag, magerr = phot.photometry_photutils(image, threshold= 5 ,check = True)
        else:
            mag, magerr = phot.photometry_SE(image, factor_aperture = 3)
        obsdate = Time(fits.getheader(image)['JD'], format = 'jd').mjd
        limmaglist.append(round(limitmag,3))
        limmagerrlist.append(round(limitmagerr,3))
        maglist.append(round(mag,3))
        magerrlist.append(round(magerr,3))
        obsdatelist.append(round(obsdate,5))
    filter_ = hdu.header['FILTER']
    os.chdir(path)
    plt.figure(figsize = (10,6))
    plt.errorbar(obsdatelist-detectiondate, maglist, yerr = magerrlist, fmt = '.', elinewidth = 1, capsize = 3, c= 'k')
    plt.grid()
    plt.xlabel(f'MJD - {detectiondate}')
    plt.ylabel('mag[AB]')
    plt.gca().invert_yaxis()
    plt.savefig(f'{path}/{obs_info["obs"][0]}_{filter_}.png')
    plt.show()

    result_tbl = Table()
    result_tbl['obsdate'] = obsdatelist
    result_tbl['UL5_4'] = limmaglist
    result_tbl['e_UL5_4'] = limmagerrlist
    result_tbl['mag'] = maglist
    result_tbl['e_mag'] = magerrlist
    result_tbl['observatory'] = f'{obs_info["obs"][0]}'
    result_tbl['filter'] = filter_
    result_tbl['image'] = combined_imlist
    result_tbl.write(f'{path}/{obs_info["obs"][0]}_{filter_}.dat', format = 'ascii.fixed_width', overwrite = True)
    return maglist, magerrlist, obsdatelist, limmaglist, limmagerrlist



#%%

def refmake(imkey, obs_info):
    import matplotlib.pyplot as plt
    import tkinter
    from tkinter import filedialog
    import glob, os
    from multiprocessing import Process
    
    refmake = Photometry(imkey = imkey, obsinfo = obs_info)
    selected_tbl, outlier_tbl = refmake.select_outlier(remove = False, check = True)
    print(120*'-')
    choose1 = input(f'remove {len(outlier_tbl)} outlying images? (y/n)')
    if choose1 in ['Y', 'y']:
        selected_tbl, outlier_tbl = refmake.select_outlier(remove = True, check = True)
    choose2 = ''
    while choose2 not in ['Y', 'y']:
        cut_seeing = float(input('Give a cut off line for Seeing'))
        cut_depth= float(input('Give a cut off line for Depth'))
        
        all_tbl = selected_tbl
        cut_tbl = selected_tbl[(selected_tbl['Depth']>cut_depth) & (selected_tbl['Seeing'] < cut_seeing)]
        plt.scatter(all_tbl['Seeing'], all_tbl['Depth'], c= 'k', marker = 'o', s= 6, alpha = 0.6)
        plt.scatter(cut_tbl['Seeing'], cut_tbl['Depth'], c= 'r', marker = 'o', s= 6, alpha = 1, label  =f'Selected[{len(cut_tbl)}]')
        plt.axvline(x= cut_seeing, c= 'r', linestyle = '--', linewidth = 1)
        plt.axhline(y= cut_depth, c= 'r', linestyle = '--', linewidth = 1)
        plt.xlabel('Seeing[arcsec]')
        plt.ylabel('Depth[AB]')
        plt.grid()
        plt.legend()
        plt.show()
        choose3 = input(f'{len(cut_tbl)} images are selected. Do you want to check images with ds9?')
        def remove_file(path):
            file_path = 'open'
            while type(file_path)== str:
                root = tkinter.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilenames(parent=root,initialdir=path)
                for file in file_path:
                    os.system(f'rm {file}')
                    print(f'{os.path.basename(file)} removed!')
        
        if choose3 in ['y', 'Y']:
            command = 'ds9'
            path = os.path.dirname(cut_tbl['Image'][0])
            t = Process(target = remove_file, args = (path,))
            t.start()
            for image in cut_tbl['Image']:
                command = command + ' '+image
            os.system(command)
        refmake = Photometry(imkey = imkey, obsinfo = obs_info)
        cut_tbl, _ = refmake.select_outlier(remove = False, check = False)
        cut_tbl = cut_tbl[(cut_tbl['Depth']>cut_depth) & (cut_tbl['Seeing'] < cut_seeing)]
        choose2 = input(f'Make reference image with {len(cut_tbl)} images? (y/n)')
        
        if choose2 not in ['Y','y','N','n']:
            break
    
    print(60*'==')
    print(30*'=', 'Making a reference iamge', 30*'=')
    
    refmake = Photometry(imkey = imkey, obsinfo = obs_info, refimage = cut_tbl['Image'][0])
    aligned_imlist = []
    for image in cut_tbl['Image']:
        _, aligned_image = refmake.align(image, scale = True, write = True)
        aligned_imlist.append(aligned_image)
    refpath = refmake.combine_iraf(aligned_imlist)
    UPDATE_IMAGE(refpath, refmake.get_obsinfo(), refcat = 'APASS', check = True)
    print(60*'==')
    print('RESULT FILE PATH =', refpath)
    


#%%
def load_info_observatory(option, observatory, ccd, RASAmode = None):    
    from astropy.io import ascii
    allobs_info = ascii.read('/home/hhchoi1022/Desktop/Gitrepo/config/CCD.dat', format = 'fixed_width')
    obs_info = allobs_info[allobs_info['obs'] == observatory]
    if len(obs_info) >1:
        obs_info = obs_info[obs_info['ccd'] == ccd]
    if observatory == 'RASA36':
        if RASAmode in ['merge','Merge','MERGE']:
            obs_info = obs_info[obs_info['gain'] > 10]
        if RASAmode in ['high', 'High','HIGH']:
            obs_info = obs_info[obs_info['gain'] < 10]
    return obs_info

#%%
'''
    option = input('Select mode (Photometry for a designated point source with RADec / Reference image making')
    
    from astropy.io import ascii
    allobs_info = ascii.read('/home/hhchoi1022/Desktop/Gitrepo/config/CCD.dat', format = 'fixed_width')
    observatory = input(f'Select observaotry \n Options : ({set(allobs_info["obs"])})')
    obs_info = allobs_info[allobs_info['obs'] == observatory]
    if len(obs_info) >1:
        ccd = input(f'More than one CCDs found. Select one. \n {set(obs_info["ccd"])}')
        obs_info = obs_info[obs_info['ccd'] == ccd]
    if observatory == 'RASA36':
        mode = input(f'RASA36 can be operated by two different mode, choose one (High or Merge)')
        if mode in ['merge','Merge','MERGE']:
            obs_info = obs_info[obs_info['gain'] > 10]
        if mode in ['high', 'High','HIGH']:
            obs_info = obs_info[obs_info['gain'] < 10]
'''
#%%
import json
def defaultmake(existfile , configfile = 'photometry.config'):
    if existfile == True:
        print('Make default configuration file')
        configname = input('Set filename ("photometry.config")')
        option = 'photometry'
        imkey = '/data2/SN2021aefx_220627/RASA36/r/HIGH/*.fits'
        refim = '/data2/SN2021aefx_220627/RASA36/r/Ref-RASA36-NGC1566-r-3180-HIGH.com.fits'
        subtract = 'y'
        ra = '4:19:53'
        dec ='-54:56:53'
        mode = 'HIGH'
        observatory = 'RASA'
        ccd = 'KL4040'
    
        config = {"option" : option,
                  "imkey" : imkey,
                  "refim" : refim,
                  "subtract" : subtract,
                  "ra" : ra,
                  "dec" : dec,
                  "mode": mode,
                  'observatory': observatory,
                  "ccd": ccd
                  }
        json.dump(config, open(configfile,'w'), indent = 4, sort_keys = True)

def input_config(configfile):
    #configfile = '/home/hhchoi1022/Desktop/Gitrepo/photometry/photometry_KCT_r.config'
    if not os.path.isfile(configfile):
        defaultmake(True)
    
    with open(configfile,'r') as config_json:
        config =  json.load(config_json)
    return config


def run(configfile):
    config = input_config(configfile)
    option = config['option']
    imkey = config['imkey']
    refim = config['refim']
    sub = config['subtract']
    ra = config['ra']
    dec = config['dec']
    observatory = config['observatory']
    ccd = config['ccd']
    mode = config['mode']
    program = config['program']
    obs_info = load_info_observatory(option, observatory, ccd, mode)
    if sub in ['y', 'Y']:
        phot(imkey = imkey, refim = refim, obs_info = obs_info, ra = ra, dec = dec, program = program, write = True, sub = True, start = True)
    else:
        phot(imkey = imkey, refim = refim, obs_info = obs_info, ra = ra, dec = dec, program = program, write = True, sub = False, start=True)
#%%

configfilelist = glob.glob('/home/hhchoi1022/Desktop/Gitrepo/photometry/*.config')

for configfile in configfilelist:
    os.chdir('/home/hhchoi1022/Desktop/Gitrepo/photometry')
    run(configfile)
