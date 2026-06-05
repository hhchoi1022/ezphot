

#%%
if __name__ == '__main__':
    from bridge.utils import HostGalaxyCatalog
    hg = HostGalaxyCatalog()
#%%
if __name__ == '__main__':
    from bridge.connector import SQLConnector
    import numpy as np
    db_transients = SQLConnector().get_data('transient_status')
    target_transients = [
        'AT2025ajmj',
        'SN2025aftz',
        'SN2025qpk',
        'SN2025aard',
        'SN2025afih',
        'SN2025ulq'
        ]
    # Filter
    db_transients_selected = db_transients[np.isin(db_transients['objname'], target_transients)]
#%%
if __name__ == '__main__':
    from ezphot.methods.tract7dt import Tract7DTRunner
    from astropy.time import Time
    all_result_dict = dict()
        
    # for idx in range(len(db_transients_selected)):        
    idx = 4
    target_selected = str(db_transients_selected['objname'][idx])
    target_ra = db_transients_selected['ra'][idx]
    target_dec = db_transients_selected['dec'][idx]
    tile_id = db_transients_selected['tile_id'][idx]
    host_galaxy = hg.match_host(ra = target_ra, dec = target_dec, max_dell = 2.5)
    host_ra = host_galaxy['RAdeg']
    host_dec = host_galaxy['DEdeg']
    host_type = 'EXP'
    host_ellip = host_galaxy['R2'] / (host_galaxy['R1'] + host_galaxy['R2'])
    host_Re = host_galaxy['R1'] / 0.505
    host_theta = host_galaxy['PA']
    all_result_dict[target_selected] = dict()
    all_result_dict[target_selected]['target_ra'] = target_ra
    all_result_dict[target_selected]['target_dec'] = target_dec
    all_result_dict[target_selected]['host_ra'] = host_ra
    all_result_dict[target_selected]['host_dec'] = host_dec
    all_result_dict[target_selected]['host_type'] = host_type
    all_result_dict[target_selected]['host_ellip'] = host_ellip
    all_result_dict[target_selected]['host_Re'] = host_Re
    all_result_dict[target_selected]['host_theta'] = host_theta
    all_result_dict[target_selected]['tile_id'] = tile_id

    from ezphot.utils import DataBrowser
    dbrowser = DataBrowser('scidata')
    dbrowser.objname = tile_id
    target_imgset = dbrowser.search(pattern = 'coadd_scaled*com.fits', return_type = 'science')
    # target_imgset.select_images(filter = ['m400','m425','m450','m475','m500','m525','m550','m575','m600','m625','m650','m675','m700','m725','m750','m775','m800','m825','m850','m875'])
    target_imgset.target_images.sort(key = lambda x: x.filter)
    image_paths = [img.path for img in target_imgset.target_images]
    catalog_paths = [img.catalog.path for img in target_imgset.target_images]
    filter_list = [img.filter for img in target_imgset.target_images]
    target_imgset.target_images[-10].show_position(x = target_ra, y = target_dec, coord_type = 'coord', zoom_radius_pixel = 200)
    id_ = f"{target_selected}_{Time(np.nanmean([img.mjd for img in target_imgset.target_images]), format='mjd').datetime.strftime('%Y%m%d_%H%M%S')}"
    obsdate = Time(np.nanmean([img.mjd for img in target_imgset.target_images]), format='mjd').datetime.strftime('%Y%m%d_%H%M%S')
    all_result_dict[target_selected]['obsdate'] = obsdate

    target_refcatset = dbrowser.search(pattern = '*.fits.refcat', return_type = 'catalog')
    for refcat in target_refcatset.catalogs:
        refcat.data
        print(refcat.nsources)
    merged_tbl, merged_metadata = target_refcatset.merge_catalogs(join_type = 'outer')       

    import numpy as np
    from ezphot.methods.tract7dt.wrapper import Tract7DTWrapper
    self = Tract7DTRunner(image_paths = image_paths, filter_list = filter_list, catalog_paths = catalog_paths, id = id_)
    list_ra = list(merged_tbl['ra_basis']) 
    list_dec = list(merged_tbl['dec_basis']) 
    list_type = ['STAR'] * len(list_ra)
    list_ellip = [0] * len(list_ra)
    list_Re = [0] * len(list_ra)
    list_theta = [0] * len(list_ra)
    list_ra.append(target_ra)
    list_dec.append(target_dec)
    list_type.append('STAR')
    list_ellip.append(0)
    list_Re.append(0)
    list_theta.append(0)
    # list_ra.append(host_ra)
    # list_dec.append(host_dec)
    # list_type.append(host_type)
    # list_ellip.append(host_ellip)
    # list_Re.append(host_Re)
    # list_theta.append(host_theta)
    self.register_target(list_ra = list_ra, list_dec = list_dec, list_type = list_type, list_ellip = list_ellip, list_Re = list_Re, list_theta = list_theta, 
                        update_type_from_catalog = False, 
                        update_flux_from_catalog = False,
                        update_ellip_from_catalog = False, 
                        update_Re_from_catalog = True,
                        update_theta_from_catalog = True)
    self.register_reference_catalog(objname = dbrowser.objname)
    self.run()
    
    all_result_dict[target_selected]['tract7dt_result'] = self.workdir / 'final_catalog_with_fit.csv'
# %% READ TRACT7DT RESULT
if __name__ == '__main__':
    from astropy.coordinates import SkyCoord
    from astropy.io import ascii
    idx = 3
    objname = target_transients[idx]
    tract7DT_result = all_result_dict[objname]['tract7dt_result']
    target_ra = all_result_dict[objname]['target_ra']
    target_dec = all_result_dict[objname]['target_dec']
    host_ra = all_result_dict[objname]['host_ra']
    host_dec = all_result_dict[objname]['host_dec']
    host_type = all_result_dict[objname]['host_type']
    host_ellip = all_result_dict[objname]['host_ellip']
    host_Re = all_result_dict[objname]['host_Re']
    host_theta = all_result_dict[objname]['host_theta']
    tile_id = all_result_dict[objname]['tile_id']
    target_coord = SkyCoord(target_ra, target_dec, unit = 'deg')
    obsdate = all_result_dict[objname]['obsdate']
    result = ascii.read(str(tract7DT_result))
    flux_keys = [key for key in result.colnames if key.startswith('FLUX_') and not key.endswith('_fit')]
    filter_keys = []
    for flux_key in flux_keys:
        mag_key = flux_key.replace('FLUX_', 'MAG_')
        flux_key_fit = flux_key + '_fit'
        fluxerr_key_fit = flux_key.replace('FLUX_', 'FLUXERR_') + '_fit'
        mag_key_fit = mag_key + '_fit'
        magerr_key_fit = mag_key_fit.replace('MAG_', 'MAGERR_')
        filter = flux_key.replace('FLUX_', '')
        filter_keys.append(filter)
        flux_input = result[flux_key]
        flux_fit = result[flux_key_fit]    
        fluxerr_fit = result[fluxerr_key_fit]
        mag_fit = -2.5*np.log10(flux_fit)
        magerr_fit = 2.5/np.log(10) * fluxerr_fit / flux_fit
        result[mag_key_fit] = mag_fit
        result[magerr_key_fit] = magerr_fit
    result_from_ezphot = result
    result_from_tract7dt = result
#%% MATCH PSF AND REF CATALOGS
if __name__ == '__main__':
    from ezphot.helper import Helper
    from astropy.coordinates import SkyCoord
    from ezphot.skycatalog import SkyCatalog
    refcatalog = SkyCatalog(objname = tile_id)
    ref_tbl = refcatalog.data
    result_from_ezphot.sort('MAGERR_m400_fit')
    result_from_tract7dt.sort('MAGERR_m400_fit')

    max_distance_arcsec = 10
    helper = Helper()
    ra_ezphot = result_from_ezphot['RA']
    dec_ezphot = result_from_ezphot['DEC']
    ra_tract7dt = result_from_tract7dt['RA']
    dec_tract7dt = result_from_tract7dt['DEC']
    skycoord_ref = SkyCoord(ref_tbl['ra'], ref_tbl['dec'], unit = 'deg')
    skycoord_ezphot = SkyCoord(ra_ezphot, dec_ezphot, unit = 'deg')
    skycoord_tract7dt = SkyCoord(ra_tract7dt, dec_tract7dt, unit = 'deg')
    idx_ezphot, idx_ref_ezphot, _ = helper.cross_match(skycoord_ezphot, skycoord_ref, max_distance_second = max_distance_arcsec)
    idx_tract7dt, idx_ref_tract7dt, _ = helper.cross_match(skycoord_tract7dt, skycoord_ref, max_distance_second = max_distance_arcsec)
    result_matched_ezphot = result_from_ezphot[idx_ezphot]
    ref_tbl_matched_ezphot = ref_tbl[idx_ref_ezphot]
    result_matched_tract7dt = result_from_tract7dt[idx_tract7dt]
    ref_tbl_matched_tract7dt = ref_tbl[idx_ref_tract7dt]
#%% CALCULATE ZP FOR EACH FILTER
if __name__ == '__main__':
    helper = Helper()
    zp_all_ezphot = dict()
    mag_psf_all_ezphot = dict()
    magerr_psf_all_ezphot = dict()
    mag_ref_all_ezphot = dict()
    zp_all_tract7dt = dict()
    mag_psf_all_tract7dt = dict()
    magerr_psf_all_tract7dt = dict()
    mag_ref_all_tract7dt = dict()
    for flux_key in flux_keys:
        filter = flux_key.replace('FLUX_', '')
        
        fluxerr_key_fit = flux_key.replace('FLUX_', 'FLUXERR_') + '_fit'
        mag_key_fit = flux_key.replace('FLUX_', 'MAG_') + '_fit'
        magerr_key_fit = mag_key_fit.replace('MAG_', 'MAGERR_')
        mag_key_ref = f'{filter}_mag'
        
        mag_ezphot = result_matched_ezphot[mag_key_fit]
        magerr_ezphot = result_matched_ezphot[magerr_key_fit]
        mag_ref_ezphot = ref_tbl_matched_ezphot[mag_key_ref]
        mag_tract7dt = result_matched_tract7dt[mag_key_fit]
        magerr_tract7dt = result_matched_tract7dt[magerr_key_fit]
        mag_ref_tract7dt = ref_tbl_matched_tract7dt[mag_key_ref]
        zp_all_ezphot[filter] = mag_ref_ezphot - mag_ezphot
        mag_psf_all_ezphot[filter] = mag_ezphot
        magerr_psf_all_ezphot[filter] = magerr_ezphot
        mag_ref_all_ezphot[filter] = mag_ref_ezphot
        zp_all_tract7dt[filter] = mag_ref_tract7dt - mag_tract7dt
        mag_psf_all_tract7dt[filter] = mag_tract7dt
        magerr_psf_all_tract7dt[filter] = magerr_tract7dt
        mag_ref_all_tract7dt[filter] = mag_ref_tract7dt
#%% PLOT ZP FOR EACH FILTER
from astropy.stats import sigma_clipped_stats
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ezphot.dataobjects import LightCurve
    zp_filter_ezphot = dict()
    zp_filter_tract7dt = dict()
    zp_filter_std_ezphot = dict()
    zp_filter_std_tract7dt = dict()
    zp_filter_std = dict()
    color_map = LightCurve.FILTER_COLOR
    plt.figure(figsize = (10, 6))
    for flux_key in flux_keys:
        filter_ = flux_key.replace('FLUX_', '')
        mag_psf_ezphot = mag_psf_all_ezphot[filter_]
        mag_psf_tract7dt = mag_psf_all_tract7dt[filter_]
        mag_ref_ezphot = mag_ref_all_ezphot[filter_]
        mag_ref_tract7dt = mag_ref_all_tract7dt[filter_]
        zp_ezphot = zp_all_ezphot[filter_]
        zp_tract7dt = zp_all_tract7dt[filter_]
        zp_ezphot = zp_all_ezphot[filter_]
        zp_tract7dt = zp_all_tract7dt[filter_]
        
        zp_mean_ezphot, zp_median_ezphot, zp_std_ezphot = sigma_clipped_stats(zp_ezphot, sigma = 3, maxiters = 5)
        zp_mean_tract7dt, zp_median_tract7dt, zp_std_tract7dt = sigma_clipped_stats(zp_tract7dt, sigma = 3, maxiters = 5)
        
        if filter_ == 'm875':
            plt.scatter(mag_psf_tract7dt + zp_median_tract7dt, zp_tract7dt, edgecolor = 'k', facecolor = 'none', marker = 'D', label = f'{filter_}[ZP [GaiaXP_selected, Tract7DT] = {zp_median_tract7dt:.2f} ± {zp_std_tract7dt:.2f}]')
            plt.axhline(zp_median_tract7dt, color = color_map[filter_], ls = '--', lw = 1)
            plt.scatter(mag_psf_ezphot + zp_median_ezphot, zp_ezphot, edgecolor = 'r', facecolor = 'none', marker = 'D', label = f'{filter_}[ZP = {zp_median_ezphot:.2f} ± {zp_std_ezphot:.2f}]')
            plt.axhline(zp_median_ezphot, color = color_map[filter_], ls = '--', lw = 1)

        zp_filter_ezphot[filter_] = zp_median_ezphot
        zp_filter_tract7dt[filter_] = zp_median_tract7dt
        zp_filter_std_ezphot[filter_] = zp_std_ezphot
        zp_filter_std_tract7dt[filter_] = zp_std_tract7dt
    plt.legend(ncols = 3)
    plt.ylabel(r'ZP [M$_{PSF}$ - M$_{GAIAXP}$]', fontsize = 14)
    plt.xlabel('Magnitude [mag]', fontsize = 14)
    #plt.ylim(24, 26)
    plt.show()
#%% COMPARE WITH EZPHOT RESULT
if __name__ == '__main__':
    dbrowser = DataBrowser('scidata')
    dbrowser.objname = tile_id
    catalogset = dbrowser.search(pattern = 'coadd*circ*.cat', return_type = 'catalog')
    catalogset.select_sources(target_ra, target_dec)
    from ezphot.dataobjects import PhotometricSpectrum
    catalogset.select_catalogs(filter = filter_list)
    photspec = PhotometricSpectrum(catalogset)
    flux_key_ezphot = 'MAGSKY_APER_2'
    fluxerr_key_ezphot = 'MAGERR_APER_2'
    zperr_key_ezphot = 'ZPERR_APER_2'
    photspec_result = photspec.plot(ra = target_ra, dec = target_dec, flux_key = flux_key_ezphot, fluxerr_key = fluxerr_key_ezphot)
    photspec_fig = list(photspec_result[0].values())[0]
    photspec_tbl = photspec_result[3]
    photspec_ax = list(photspec_result[2].values())[0]
    target_img = catalogset.target_catalogs[5].target_img
#%%
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize = (10, 6))
    # ax.step(spec.wavelength.nm, spec.ab.flux.value, color = 'k', alpha = 0.2)
    result_target = result[SkyCoord(result['RA'], result['DEC'], unit = 'deg').separation(target_coord).value < 1/3600][0]
    magdiff_all = []
    magdiff_all_spec = []
    for flux_key in flux_keys:
        try:
            filter_key = flux_key.replace('FLUX_', '')
            fluxerr_key = flux_key.replace('FLUX_', 'FLUXERR_')
            flux_key_fit = flux_key + '_fit'
            fluxerr_key_fit = fluxerr_key + '_fit'
            filter = flux_key.replace('FLUX_', '')
             
            
            flux_fit = result_target[flux_key_fit]
            fluxerr_fit = result_target[fluxerr_key_fit]
            abmag_fit = -2.5*np.log10(flux_fit) + zp_filter_ezphot[filter]
            abmagerr_fit = 2.5/np.log(10) * fluxerr_fit / flux_fit
            zperr_fit = zp_filter_std_ezphot[filter]
            magerr_fit = np.sqrt(abmagerr_fit**2 + zperr_fit**2)
            
            row_photspec = photspec_tbl[photspec_tbl['filter'] == filter_key]
            abmag_photspec = row_photspec[flux_key_ezphot]
            magerr_photspec = np.sqrt(row_photspec[fluxerr_key_ezphot]**2 + row_photspec[zperr_key_ezphot]**2)
            
            wl = photspec.FILTER_PIVOT_WAVELENGTH_NM[filter_key]
            
            # Plot HOTPANTS
            ax.scatter(wl, abmag_photspec, edgecolor = 'blue', facecolor = 'none', marker = 'D')
            ax.errorbar(wl, abmag_photspec, yerr = magerr_photspec, color = 'blue', fmt = 'none')
            
            # Plot Tract7DT
            ax.scatter(wl, abmag_fit, edgecolor = 'red', facecolor = 'none', marker = 'D')
            ax.errorbar(wl, abmag_fit, yerr = magerr_fit, color = 'red', fmt = 'none')
        except Exception as e:
            print(e)
            pass
    # Inverse the y-axis
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    
    ax.scatter(0, 0 , edgecolor = 'red', facecolor = 'none', marker = 'D', label = f'Tract7DT')
    ax.scatter(0, 0 , edgecolor = 'blue', facecolor = 'none', marker = 'D', label = f'EZPHOT')
    ax.set_xlim(350, 1000)
    ax.set_ylim(ylim)
    ax.legend(loc = 'upper left')
#%%
if __name__ == '__main__':
    catalogset = dbrowser.search(pattern = 'coadd_scaled*.fits.cat', return_type = 'catalog')
    catalogset.select_sources(target_ra, target_dec)
    from ezphot.dataobjects import PhotometricSpectrum
    catalogset.select_catalogs(filter = filter_list)
    photspec = PhotometricSpectrum(catalogset)
    flux_key_ezphot = 'MAGSKY_APER_2'
    fluxerr_key_ezphot = 'MAGERR_APER_2'
    zperr_key_ezphot = 'ZPERR_APER_2'
    photspec_result = photspec.plot(ra = target_ra, dec = target_dec, flux_key = flux_key_ezphot, fluxerr_key = fluxerr_key_ezphot, matching_radius_arcsec = 10)
    photspec_fig = list(photspec_result[0].values())[0]
    photspec_tbl = photspec_result[3]
    photspec_ax = list(photspec_result[2].values())[0]
# %%    

if __name__ == '__main__':
    from ezphot.dataobjects import Spectrum
    idx = 0

    specfiles = glob.glob(f'/home/hhchoi1022/code/*{objname}*')
    specfile = specfiles[idx]
    spec = Spectrum(specfile)
    spec.show(show_flux_unit =  'ab')
    synphot_dict = spec.synphot(filterset = ['medium', 'r'], visualize = False, visualize_transmission = False, visualize_photometry_label = False)[0]
    target_coord = SkyCoord(target_ra, target_dec, unit = 'deg')
    result_target = result[SkyCoord(result['RA'], result['DEC'], unit = 'deg').separation(target_coord).value < 10/3600]
    all_filter_keys = list(synphot_dict.keys())
    mag_offsets_ezphot = dict()
    mag_offsets_tract7dt = dict()
    for filter_key in all_filter_keys:
        flux_key = f'FLUX_{filter_key}'
        fluxerr_key = flux_key.replace('FLUX_', 'FLUXERR_')
        flux_key_fit = flux_key + '_fit'
        fluxerr_key_fit = fluxerr_key + '_fit'
        
        row_photspec = photspec_tbl[photspec_tbl['filter'] == filter_key]
        if len(row_photspec) == 0:
            pass
        else:
            mag_offset_ezphot = synphot_dict[filter_key]['mag'] - row_photspec[flux_key_ezphot][0]
            mag_offsets_ezphot[filter_key] = mag_offset_ezphot
        if flux_key_fit in result_target.colnames:
            mag_tract7dt = -2.5*np.log10(result_target[flux_key_fit][0]) + zp_filter_ezphot[filter_key]
            mag_offset_tract7dt = synphot_dict[filter_key]['mag'] - mag_tract7dt
            mag_offsets_tract7dt[filter_key] = mag_offset_tract7dt
    mean_ezphot, median_ezphot, std_ezphot = sigma_clipped_stats(list(mag_offsets_ezphot.values()), sigma = 3, maxiters = 5)
    mean_tract7dt, median_tract7dt, std_tract7dt = sigma_clipped_stats(list(mag_offsets_tract7dt.values()), sigma = 3, maxiters = 5)
    print(f'Mag offset mean: {mean_ezphot:.2f}, std: {std_ezphot:.2f}, median: {median_ezphot:.2f}')
    print(f'Mag offset mean: {mean_tract7dt:.2f}, std: {std_tract7dt:.2f}, median: {median_tract7dt:.2f}')
#%%
if __name__ == '__main__':
    import glob
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.step(spec.wavelength.nm, spec.ab.flux.value, color = 'k', alpha = 0.2)
    result_target = result[SkyCoord(result['RA'], result['DEC'], unit = 'deg').separation(target_coord).value < 1/3600][0]
    result_host = result[result['TYPE'] == 'EXP'][0]
    mag_all = []
    magdiff_all_spec = []
    for flux_key in flux_keys:
        try:
            filter_key = flux_key.replace('FLUX_', '')
            fluxerr_key = flux_key.replace('FLUX_', 'FLUXERR_')
            flux_key_fit = flux_key + '_fit'
            fluxerr_key_fit = fluxerr_key + '_fit'
            filter = flux_key.replace('FLUX_', '')
             
            
            flux_fit = result_target[flux_key_fit]
            fluxerr_fit = result_target[fluxerr_key_fit]
            flux_host = result_host[flux_key_fit]
            fluxerr_host = result_host[fluxerr_key_fit]
            abmag_fit = -2.5*np.log10(flux_fit) + zp_filter_ezphot[filter]
            abmag_host = -2.5*np.log10(flux_host) + zp_filter_ezphot[filter]
            abmagerr_fit = 2.5/np.log(10) * fluxerr_fit / flux_fit
            abmagerr_host = 2.5/np.log(10) * fluxerr_host / flux_host
            zperr_fit = zp_filter_std_ezphot[filter]
            magerr_fit = np.sqrt(abmagerr_fit**2 + zperr_fit**2)
            magerr_host = np.sqrt(abmagerr_host**2 + zperr_fit**2)
            mag_all.append(abmag_fit)
            
            row_photspec = photspec_tbl[photspec_tbl['filter'] == filter_key]
            abmag_photspec = row_photspec[flux_key_ezphot]
            magerr_photspec = np.sqrt(row_photspec[fluxerr_key_ezphot]**2 + row_photspec[zperr_key_ezphot]**2)
            
            wl = photspec.FILTER_PIVOT_WAVELENGTH_NM[filter_key]
            
            # Plot spectrum
            ax.scatter(wl, synphot_dict[filter_key]['mag'], edgecolor = 'black', facecolor = 'none', marker = 'D')

            # Plot HOTPANTS
            mag_diff = float(synphot_dict[filter_key]['mag'] - abmag_photspec[0] - mean_ezphot)
            ax.scatter(wl, abmag_photspec + mean_ezphot, edgecolor = 'blue', facecolor = 'none', marker = 'D')
            ax.errorbar(wl, abmag_photspec + mean_ezphot, yerr = magerr_photspec, color = 'blue', fmt = 'none')
            # ax.text(wl, 16.4, f'{mag_diff:.3f}', color = 'b', rotation = 90)
            
            # Plot Tract7DT
            magdiff_between_synphot_and_tract7dt = float(synphot_dict[filter_key]['mag'] - abmag_fit - mean_tract7dt)
            ax.scatter(wl, abmag_fit + mean_tract7dt, edgecolor = 'red', facecolor = 'none', marker = 'D')
            ax.errorbar(wl, abmag_fit + mean_tract7dt, yerr = magerr_fit, color = 'red', fmt = 'none')
            # ax.text(wl, 17.2, f'{magdiff_between_synphot_and_tract7dt:.3f}', color = 'r', rotation = 90)
        except:
            pass

    ax.scatter(0, 0 , edgecolor = 'k', facecolor = 'none', marker = 'D', label = f'Spectrum ({spec.obsdate})')
    ax.scatter(0, 0 , edgecolor = 'red', facecolor = 'none', marker = 'D', label = f'Tract7DT + {mean_tract7dt:.2f} ({obsdate})')
    ax.scatter(0, 0 , edgecolor = 'blue', facecolor = 'none', marker = 'D', label = f'Original + {mean_ezphot:.2f} ({obsdate})')

    # ax.set_ylim([17.4, 16.0])
    ax.set_xlim(350, 1000)
    # ax.set_ylim(np.nanmedian(mag_all) -1 , np.nanmedian(mag_all) + 3)
    ax.set_ylim(16,10)
    ax.legend(loc = 'upper left')
# %%
print(objname)
print(result_target['patch_tag'])
fig_path = tract7DT_result.parent / 'tractor_out_patches' / f'{result_target["patch_tag"]}' / 'cutouts' /  f'src_{result_target["ID"]}.png'
fig_path_host = tract7DT_result.parent / 'tractor_out_patches' / f'{result_host["patch_tag"]}' / 'cutouts' /  f'src_{result_host["ID"]}.png'
# %%
fig_path
from PIL import Image
from IPython.display import display
from pathlib import Path

img = Image.open(fig_path)
display(img)
# %%
img_host = Image.open(fig_path_host)
display(img_host)
# %%