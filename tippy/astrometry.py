#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:26:17 2021

@author: hhchoi1022
"""


def astrometry(imkey, 
               scalelow=0.6, 
               scalehigh=0.8, 
               verwrite=False,
               ):
    """
    1. Description
    : Solving WCS coordinates using Astrometry.net software. For better performance in especially B band images, --use-sextractor mode is added. This mode needs SExtractor configuration files. So please posit configuration files for your working directory. cpulimit 300 is also added to prevent too long processing time for bad images.
    : scalelow and scalehigh for the range of pixscale estimation
    2. Usage
    >>> astrometry()

    3. History
    2018.03    Created by G.Lim.
    2018.12.18 Edited by G.Lim. SExtractor mode is added.
    2018.12.21 Edited by G.Lim. Define SAO_astrometry function.
    2020.03.01 --backend-config is added to have the system find INDEX files.
    2021.12.29 Edited by HH.Choi.  
    """
    import os,sys
    import glob
    import subprocess
    import numpy as np
    
    imagelist = glob.glob(imkey)
    imagelist_name = [os.path.basename(name) for name in imagelist]
    imagelist.sort()
    imagelist_name.sort()
    
    ###### SExtractor configuration file here ######
    sexdir = '/data2/sextractor'
    os.chdir(sexdir)
    sexconfig = f'{sexdir}/astrometry.SEconfig'
    os.system(f'cp {sexconfig} {sexdir}/astrometry.SEparam {sexdir}/*.conv {sexdir}/*.nnw {os.path.dirname(imkey)}')
    ################################################
    os.chdir(os.path.dirname(imkey))
    print('Solving WCS using Astrometry')
    for i, image in enumerate(imagelist):
        if overwrite == False :
            com=f'solve-field {image} --cpulimit 60 --overwrite --use-sextractor  --sextractor-config {sexconfig} --x-column X_IMAGE --y-column Y_IMAGE --sort-column MAG_AUTO --sort-ascending --scale-unit arcsecperpix --scale-low {str(scalelow)} --scale-high {str(scalehigh)} --no-remove-lines --uniformize 0 --no-plots --radius 0.5 --new-fits a{imagelist_name[i]} --temp-dir .\n'
        elif overwrite == True :
            com=f'solve-field {image}  --cpulimit 60 --overwrite --use-sextractor  --sextractor-config {sexconfig} --x-column X_IMAGE --y-column Y_IMAGE --sort-column MAG_AUTO --sort-ascending --scale-unit arcsecperpix --scale-low {str(scalelow)}--scale-high {str(scalehigh)}  --no-remove-lines --uniformize 0 --no-plots --radius 0.5 --new-fits {imagelist_name[i]} --overwrite --temp-dir .\n'
        print(com)
        print(str(i)+' th of '+str(len(imagelist)))
        os.system(com)
    orinum = subprocess.check_output(f'ls C*.fits | wc -l', shell=True)
    resnum = subprocess.check_output(f'ls a*.fits | wc -l', shell=True)
    print("from "+str(orinum[:-1])+" files , "+str(resnum[:-1])+" files are solved.")
    print("All done.")
    os.system('rm tmp*')
    os.system('rm astrometry*')
    os.system('rm *.conv')
    os.system('rm default.nnw')
    os.system('rm *.wcs *.rdls *.corr *.xyls *.solved *.axy *.match ')
    print('Astrometry process is complete.')
