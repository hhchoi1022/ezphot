
# %%
if __name__ == '__main__':
    from bridge.utils import HostGalaxyCatalog
    hg = HostGalaxyCatalog()
#%%
if __name__ == '__main__':
    from ezphot.utils import DataBrowser
    dbrowser = DataBrowser('scidata')
    dbrowser.objname = 'T20307'
    target_imgset = dbrowser.search(pattern = 'coadd**.fits', return_type = 'science')
    target_imgset.select_images(filter = ['m400','m425','m450','m475','m500','m525','m550','m575','m600','m625','m650','m675','m700','m725','m750','m775','m800','m825','m850','m875'])
    target_imgset.target_images.sort(key = lambda x: x.filter)
    image_paths = [img.path for img in target_imgset.target_images]
    catalog_paths = [img.catalog.path for img in target_imgset.target_images]
    filter_list = [img.filter for img in target_imgset.target_images]
#%%
# ================================================
# ZP CALIBRATION
# ================================================
#%% FIND REFERENCE CATALOG
if __name__ == '__main__':
    target_refcatset = dbrowser.search(pattern = 'coadd*.fits.refcat', return_type = 'catalog')
    for refcat in target_refcatset.catalogs:
        refcat.data
        print(refcat.nsources)
    merged_tbl, merged_metadata = target_refcatset.merge_catalogs()      
#%%
merged_tbl = merged_tbl[merged_tbl['n_detections'] > 3] 
merged_tbl  = target_refcatset.catalogs[8].data
# %% RUN TRACT7DT
if __name__ == '__main__':
    target_ra = [30.7663010688] # SN2025afih
    target_dec = [4.24367725701]
    # target_ra = [101.796646] # AT2025hp
    # target_dec = [-10.420567]
    target_type = ['STAR']
    host_tbl = hg.search_catalog(ra = target_ra[0], dec = target_dec[0], radius_deg = 60/3600)
    if len(host_tbl) > 0:
        host_ra = host_tbl['RA'][0]
        host_dec = host_tbl['Dec'][0]
        host_type = 'EXP'
        target_ra.append(host_ra)
        target_dec.append(host_dec)
        target_type.append(host_type)        
#%%
if __name__ == '__main__':
    from ezphot.utils.tract7dt import Tract7DTRunner
    import numpy as np
    self = Tract7DTRunner(image_paths = image_paths, filter_list = filter_list, catalog_paths = catalog_paths)
    list_ra = list(merged_tbl['X_WORLD']) 
    list_dec = list(merged_tbl['Y_WORLD']) 
    list_type = ['STAR'] * len(list_ra)
    list_ra.extend(target_ra)
    list_dec.extend(target_dec)
    list_type.extend(target_type)
    self.register_target(list_ra = list_ra, list_dec = list_dec, list_type = list_type, update_type_from_catalog = False, update_flux_from_catalog = False)
    self.register_reference_catalog(objname = dbrowser.objname)
    self.run()
# %% READ TRACT7DT RESULT
if __name__ == '__main__':
    from astropy.io import ascii
    result = ascii.read(self.workdir / 'final_catalog_with_fit.csv')
    result = result[:19]
    flux_keys = [key for key in result.colnames if key.startswith('FLUX_') and not key.endswith('_fit')]
    filter_keys = []
    for flux_key in flux_keys:
        mag_key = flux_key.replace('FLUX_', 'MAG_')
        flux_key_fit = flux_key + '_fit'
        mag_key_fit = mag_key + '_fit'
        filter = flux_key.replace('FLUX_', '')
        filter_keys.append(filter)
        flux_input = result[flux_key]
        flux_fit = result[flux_key_fit]    
        mag_fit = -2.5*np.log10(flux_fit)
        result[mag_key_fit] = mag_fit
#%% MATCH PSF AND REF CATALOGS
if __name__ == '__main__':
    from ezphot.helper import Helper
    from astropy.coordinates import SkyCoord
    from ezphot.skycatalog import SkyCatalog
    refcatalog = SkyCatalog(objname = dbrowser.objname)
    ref_tbl = refcatalog.data

    max_distance_arcsec = 10
    helper = Helper()
    ra_psf = result['RA']
    dec_psf = result['DEC']
    skycoord_psf = SkyCoord(ra_psf, dec_psf, unit = 'deg')
    ra_ref = ref_tbl['ra']
    dec_ref = ref_tbl['dec']
    skycoord_ref = SkyCoord(ra_ref, dec_ref, unit = 'deg')
    idx_psf, idx_ref, _ = helper.cross_match(skycoord_psf, skycoord_ref, max_distance_second = max_distance_arcsec)
    result_matched = result[idx_psf]
    ref_tbl_matched = ref_tbl[idx_ref]
#%% CALCULATE ZP FOR EACH FILTER
if __name__ == '__main__':
    helper = Helper()
    zp_all = dict()
    mag_psf_all = dict()
    mag_ref_all = dict()
    for flux_key in flux_keys:
        mag_key_fit = flux_key.replace('FLUX_', 'MAG_') + '_fit'
        filter = flux_key.replace('FLUX_', '')
        mag_key_ref = f'{filter}_mag'
        mag_from_psf = result_matched[mag_key_fit]
        mag_from_ref = ref_tbl_matched[mag_key_ref]
        zp_all[filter] = mag_from_ref - mag_from_psf
        mag_psf_all[filter] = mag_from_psf
        mag_ref_all[filter] = mag_from_ref
#%% PLOT ZP FOR EACH FILTER
from astropy.stats import sigma_clipped_stats
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ezphot.dataobjects import LightCurve
    zp_filter = dict()
    zp_filter_std = dict()
    color_map = LightCurve.FILTER_COLOR
    plt.figure(figsize = (10, 6))
    for filter_ in filter_list:
        mag_psf = mag_psf_all[filter_]
        mag_ref = mag_ref_all[filter_]
        zp = zp_all[filter_]
        zp_median = np.median(zp)
        zp_std = np.std(zp)
        from astropy.stats import SigmaClip
        sc = SigmaClip(sigma = 3, maxiters = 10)
        zp_clean_sigclip = sc(zp)
        mag_psf_clean = mag_psf[~zp_clean_sigclip.mask]
        mag_ref_clean = mag_ref[~zp_clean_sigclip.mask]
        zp_clean = zp[~zp_clean_sigclip.mask]
        zp_median_clean = np.median(zp_clean)
        zp_std_clean = np.std(zp_clean)

        # zp_mean, zp_median, zp_std = sigma_clipped_stats(zp, sigma = 3, maxiters = 5)
        plt.scatter(mag_psf + zp_median, zp, edgecolor = color_map[filter_], facecolor = 'none', marker = 'D', alpha = 0.1)
        plt.scatter(mag_psf_clean + zp_median, zp_clean, edgecolor = color_map[filter_], facecolor = 'none', marker = 'D', label = f'{filter_}[ZP = {zp_median_clean:.2f} ± {zp_std_clean:.2f}]')
        plt.axhline(zp_median, color = color_map[filter_], ls = '--', lw = 1)
        # plt.axhline(zp_median + zp_std, color = 'k', ls = '--', lw = 1)
        # plt.axhline(zp_median - zp_std, color = 'k', ls = '--', lw = 1)
        # plt.text(0.05, 0.90, f'ZP = {zp_median:.3f} ± {zp_std:.3f}', transform = plt.gca().transAxes, fontsize = 14)
        # plt.ylim(zp_median - 0.5, zp_median + 0.5)
        # plt.show()    
        zp_filter[filter_] = zp_median
        zp_filter_std[filter_] = zp_std
    plt.legend(ncols = 3)
    plt.ylabel(r'ZP [M$_{PSF}$ - M$_{GAIAXP}$]', fontsize = 14)
    plt.xlabel('Magnitude [mag]', fontsize = 14)
    #plt.ylim(24, 26)
    plt.show()
#%% COMPARE WITH EZPHOT RESULT
if __name__ == '__main__':
    catalogset = dbrowser.search(pattern = 'coadd**.fits.cat', return_type = 'catalog')
    i = 9
    ra = result['RA'][i]
    dec = result['DEC'][i]
    catalogset.select_sources(ra, dec)
    from ezphot.dataobjects import PhotometricSpectrum
    photspec = PhotometricSpectrum(catalogset)
    flux_key_ezphot = 'MAGSKY_APER_2'
    fluxerr_key_ezphot = 'MAGERR_APER_2'
    zperr_key_ezphot = 'ZPERR_APER_2'
    photspec_result = photspec.plot(ra = ra, dec = dec, flux_key = flux_key_ezphot, fluxerr_key = fluxerr_key_ezphot)
    photspec_fig = list(photspec_result[0].values())[0]
    photspec_tbl = photspec_result[3]
    photspec_ax = list(photspec_result[2].values())[0]
# %%
if __name__ == '__main__':
    result_target = result[i]
    result_host = result[i]
    magdiff_all = []
    for flux_key in flux_keys:
        filter_key = flux_key.replace('FLUX_', '')
        fluxerr_key = flux_key.replace('FLUX_', 'FLUXERR_')
        flux_key_fit = flux_key + '_fit'
        fluxerr_key_fit = fluxerr_key + '_fit'
        filter = flux_key.replace('FLUX_', '')
        
        flux_fit = result_target[flux_key_fit]
        fluxerr_fit = result_target[fluxerr_key_fit]
        abmag_fit = -2.5*np.log10(flux_fit) + zp_filter[filter]
        abmagerr_fit = 2.5/np.log(10) * fluxerr_fit / flux_fit
        zperr_fit = zp_filter_std[filter]
        magerr_fit = np.sqrt(abmagerr_fit**2 + zperr_fit**2)
        
        row_photspec = photspec_tbl[photspec_tbl['filter'] == filter_key]
        abmag_photspec = row_photspec[flux_key_ezphot]
        magerr_photspec = np.sqrt(row_photspec[fluxerr_key_ezphot]**2 + row_photspec[zperr_key_ezphot]**2)
        
        wl = photspec.FILTER_PIVOT_WAVELENGTH_NM[filter_key]
        mag_diff = float(abmag_fit - abmag_photspec[0])
        photspec_ax.scatter(wl, abmag_fit, edgecolor = 'red', facecolor = 'none', marker = 'D')
        photspec_ax.errorbar(wl, abmag_fit, yerr = magerr_fit, color = 'r', fmt = 'none')
        photspec_ax.text(wl, 15.0, f'{mag_diff:.3f}', color = 'r', rotation = 90)
        magdiff_all.append(mag_diff)
    print(np.mean(magdiff_all), np.std(magdiff_all), np.median(magdiff_all))
#%%
photspec_fig
#%%
# # ================================================
# # Photometry
# # ================================================
# # RUN TRACT7DT (TARGET)
# if __name__ == '__main__':
#     import numpy as np
#     self = Tract7DTRunner(image_paths = image_paths, filter_list = filter_list, catalog_paths = catalog_paths)
#     list_ra = [233.857430764]
#     list_dec = [12.0577222937]
#     list_type = ['STAR'] * len(list_ra)
#     self.register_target(list_ra = list_ra, list_dec = list_dec, list_type = list_type, update_type_from_catalog = False)
#     self.register_reference_catalog(objname = 'T22956')
#     self.run()

# #%% READ TRACT7DT RESULT
# if __name__ == '__main__':
#     from astropy.io import ascii
#     result = ascii.read(self.workdir / 'final_catalog_with_fit.csv')    
# #%% COMPARE WITH EZPHOT RESULT
# if __name__ == '__main__':
#     catalogset = dbrowser.search(pattern = 'coadd*20250520*.fits.cat', return_type = 'catalog')
#     catalogset.select_sources(233.857430764, 12.0577222937)
#     from ezphot.dataobjects import PhotometricSpectrum
#     photspec = PhotometricSpectrum(catalogset)
#     flux_key_ezphot = 'MAGSKY_APER_2'
#     fluxerr_key_ezphot = 'MAGERR_APER_2'
#     zperr_key_ezphot = 'ZPERR_APER_2'
#     photspec_result = photspec.plot(ra = 233.857430764, dec = 12.0577222937, flux_key = flux_key_ezphot, fluxerr_key = fluxerr_key_ezphot)
#     photspec_fig = photspec_result[0]['2025-05-20 02:05']
#     photspec_tbl = photspec_result[3]
#     photspec_ax = photspec_result[2]['2025-05-20 02:05']
# # %%
# if __name__ == '__main__':
#     magdiff_all = []
#     chisq_all = []
#     for flux_key in flux_keys:
#         filter_key = flux_key.replace('FLUX_', '')
#         flux_key_fit = flux_key + '_fit'
#         filter = flux_key.replace('FLUX_', '')
#         flux_fit = result[flux_key_fit]
#         abmag_fit = -2.5*np.log10(flux_fit) + zp_filter[filter]
#         row_photspec = photspec_tbl[photspec_tbl['filter'] == filter_key]
#         abmag_photspec = row_photspec[flux_key_ezphot]
#         magerr_photspec = np.sqrt(row_photspec[fluxerr_key_ezphot]**2 + row_photspec[zperr_key_ezphot]**2)
#         chisq = (abmag_fit[0] - abmag_photspec[0])**2 / magerr_photspec[0]**2
#         wl = photspec.FILTER_PIVOT_WAVELENGTH_NM[filter_key]
#         mag_diff = float(abmag_fit[0] - abmag_photspec[0])
#         photspec_ax.scatter(wl, abmag_fit[0], edgecolor = 'blue', facecolor = 'none', marker = 'D')
#         photspec_ax.text(wl, 14.3, f'{mag_diff:.3f}', color = 'blue', rotation = 90)
#         magdiff_all.append(mag_diff)
#         chisq_all.append(chisq)