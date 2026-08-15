



# Check difference FWHM values between Source-extractor catalog and Gaussian FWHM
#%%
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.objname = 'T02080'
target_imgset = dbrowser.search(pattern = '7DT*20260419*m475*0000.fits', return_type = 'science')
target_img_m475 = target_imgset.target_images[0]
target_imgset = dbrowser.search(pattern = '7DT*20260419*m500*0000.fits', return_type = 'science')
target_img_m500 = target_imgset.target_images[0]
target_imgset = dbrowser.search(pattern = '7DT*20260419*m525*0000.fits', return_type = 'science')
target_img_m525 = target_imgset.target_images[0]
#%%
import numpy as np
mag_m475_sex = target_img_m475.refcatalog.data['MAGSKY_AUTO']
fwhm_m475_sex = 3600 * target_img_m475.refcatalog.data['FWHM_WORLD']
mag_m500_sex = target_img_m500.refcatalog.data['MAGSKY_AUTO']
fwhm_m500_sex = 3600 * target_img_m500.refcatalog.data['FWHM_WORLD']
mag_m525_sex = target_img_m525.refcatalog.data['MAGSKY_AUTO']
fwhm_m525_sex = 3600 * target_img_m525.refcatalog.data['FWHM_WORLD']
median_fwhm_m475_sex = np.median(fwhm_m475_sex)
median_fwhm_m500_sex = np.median(fwhm_m500_sex)
median_fwhm_m525_sex = np.median(fwhm_m525_sex)
std_fwhm_m475_sex = np.std(fwhm_m475_sex)
std_fwhm_m500_sex = np.std(fwhm_m500_sex)
std_fwhm_m525_sex = np.std(fwhm_m525_sex)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize = (8,6))
plt.scatter(mag_m475_sex, fwhm_m475_sex, edgecolor = 'r', facecolor = 'none', label = f'm475 ({median_fwhm_m475_sex:.2f} ± {std_fwhm_m475_sex:.2f} arcsec)')
plt.scatter(mag_m500_sex, fwhm_m500_sex, edgecolor = 'g', facecolor = 'none', label = f'm500 ({median_fwhm_m500_sex:.2f} ± {std_fwhm_m500_sex:.2f} arcsec) ')
plt.scatter(mag_m525_sex, fwhm_m525_sex, edgecolor = 'b', facecolor = 'none', label = f'm525 ({median_fwhm_m525_sex:.2f} ± {std_fwhm_m525_sex:.2f} arcsec) ')
plt.axhline(y = median_fwhm_m475_sex, color = 'r', linestyle = '--')
plt.axhline(y = median_fwhm_m500_sex, color = 'g', linestyle = '--')
plt.axhline(y = median_fwhm_m525_sex, color = 'b', linestyle = '--')
# Fill between median +- std for each band 
xlim = plt.xlim()
plt.fill_between(xlim, median_fwhm_m475_sex - std_fwhm_m475_sex, median_fwhm_m475_sex + std_fwhm_m475_sex, color = 'r', alpha = 0.2)
plt.fill_between(xlim, median_fwhm_m500_sex - std_fwhm_m500_sex, median_fwhm_m500_sex + std_fwhm_m500_sex, color = 'g', alpha = 0.2)
plt.fill_between(xlim, median_fwhm_m525_sex - std_fwhm_m525_sex, median_fwhm_m525_sex + std_fwhm_m525_sex, color = 'b', alpha = 0.2)
plt.xlim(xlim)
plt.xlabel('mag_auto (AB)')
plt.ylabel('Source Extractor fwhm (arcsec)')
plt.legend()
plt.show()
# %%
subbkg_img_m475 = target_img_m475.subtract_background(target_img_m475.bkgmap, save = True, save_fig = True)
subbkg_img_m500 = target_img_m500.subtract_background(target_img_m500.bkgmap, save = True, save_fig = True)
subbkg_img_m525 = target_img_m525.subtract_background(target_img_m525.bkgmap, save = True, save_fig = True)
#%%
# Gaussian fitting FWHM measurement
from astropy.modeling import models, fitting

def measure_gaussian_fwhm(image_data, x_image, y_image, pixelscale, half_size = 10):
    """
    Fit 2D Gaussian + constant offset around each source position and return FWHM.

    Parameters
    ----------
    image_data : 2D np.ndarray
        Image data (background-subtracted image recommended, e.g. subbkg_img.data).
    x_image, y_image : array-like
        Source positions from SExtractor catalog (X_IMAGE, Y_IMAGE, 1-based).
    pixelscale : float
        Pixel scale in arcsec/pixel (e.g. np.mean(target_img.pixelscale)).
    half_size : int
        Half size of the fitting cutout in pixels.

    Returns
    -------
    fwhm_arcsec : np.ndarray
        Gaussian FWHM in arcsec (sqrt(fwhm_x * fwhm_y)). NaN if the fit fails.
    """
    sigma2fwhm = 2 * np.sqrt(2 * np.log(2))
    fitter = fitting.LevMarLSQFitter()
    ny, nx = image_data.shape
    yy, xx = np.mgrid[0:2 * half_size + 1, 0:2 * half_size + 1]

    fwhm_arcsec = np.full(len(x_image), np.nan)
    for i, (x, y) in enumerate(zip(x_image, y_image)):
        # SExtractor coordinates are 1-based
        x0, y0 = int(round(x - 1)), int(round(y - 1))
        if (x0 - half_size < 0 or y0 - half_size < 0 or
            x0 + half_size + 1 > nx or y0 + half_size + 1 > ny):
            continue
        cutout = image_data[y0 - half_size:y0 + half_size + 1,
                            x0 - half_size:x0 + half_size + 1]
        if not np.all(np.isfinite(cutout)):
            continue

        bkg0 = np.median(cutout)
        model = (models.Gaussian2D(amplitude = cutout.max() - bkg0,
                                   x_mean = half_size, y_mean = half_size,
                                   x_stddev = 2, y_stddev = 2)
                 + models.Const2D(amplitude = bkg0))
        try:
            fit = fitter(model, xx, yy, cutout, maxiter = 200)
        except Exception:
            continue

        sx, sy = abs(fit[0].x_stddev.value), abs(fit[0].y_stddev.value)
        fwhm_pix = sigma2fwhm * np.sqrt(sx * sy)
        # Reject unphysical fits
        if fit[0].amplitude.value <= 0 or fwhm_pix < 0.5 or fwhm_pix > 2 * half_size:
            continue
        fwhm_arcsec[i] = fwhm_pix * pixelscale
    return fwhm_arcsec

# %%
# Usage example (after subbkg_img_m475, ... are created):
# fwhm_m475_gauss = measure_gaussian_fwhm(
#     subbkg_img_m475.data,
#     target_img_m475.refcatalog.data['X_IMAGE'],
#     target_img_m475.refcatalog.data['Y_IMAGE'],
#     np.mean(subbkg_img_m475.pixelscale))

# %%
# Visual inspection of source shapes
def show_source_cutouts(image_data, x_image, y_image, mag, title = '',
                        n_sources = 10, half_size = 15, sat_level = 55000):
    """
    Show cutouts of the brightest unsaturated sources.
    Red contour = 50% of peak (half-maximum isophote):
    a round contour means near-Gaussian, elongated/flat-topped shapes mean not.
    """
    ny, nx = image_data.shape
    order = np.argsort(mag)
    ncols = 5
    nrows = int(np.ceil(n_sources / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize = (2.2 * ncols, 2.8 * nrows))
    shown = 0
    for idx in order:
        x0, y0 = int(round(x_image[idx] - 1)), int(round(y_image[idx] - 1))
        if (x0 - half_size < 0 or y0 - half_size < 0 or
            x0 + half_size + 1 > nx or y0 + half_size + 1 > ny):
            continue
        cutout = image_data[y0 - half_size:y0 + half_size + 1,
                            x0 - half_size:x0 + half_size + 1]
        if not np.all(np.isfinite(cutout)) or cutout.max() > sat_level:
            continue
        ax = axes.flat[shown]
        ax.imshow(cutout, origin = 'lower', cmap = 'gray',
                  vmin = 0, vmax = cutout.max())
        ax.contour(cutout, levels = [0.5 * cutout.max()], colors = 'r', linewidths = 1)
        ax.set_title(f'{mag[idx]:.1f} mag', fontsize = 9)
        ax.set_xticks([]); ax.set_yticks([])
        shown += 1
        if shown == n_sources:
            break
    for ax in axes.flat[shown:]:
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# %%
# show_source_cutouts(subbkg_img_m475.data,
#                     target_img_m475.refcatalog.data['X_IMAGE'],
#                     target_img_m475.refcatalog.data['Y_IMAGE'],
#                     mag_m475_sex, title = 'm475')
# show_source_cutouts(subbkg_img_m500.data,
#                     target_img_m500.refcatalog.data['X_IMAGE'],
#                     target_img_m500.refcatalog.data['Y_IMAGE'],
#                     mag_m500_sex, title = 'm500')
# show_source_cutouts(subbkg_img_m525.data,
#                     target_img_m525.refcatalog.data['X_IMAGE'],
#                     target_img_m525.refcatalog.data['Y_IMAGE'],
#                     mag_m525_sex, title = 'm525')

# %%
# PSF matching test (m500 excluded: distorted PSF)
# Convolve the smaller-seeing image with a Gaussian kernel so that its seeing
# matches the larger-seeing image. For Gaussian PSFs:
# FWHM_kernel = sqrt(FWHM_large^2 - FWHM_small^2)
from scipy.ndimage import gaussian_filter

def convolve_gaussian(image_data, kernel_fwhm_arcsec, pixelscale):
    """
    Convolve the whole image with a circular Gaussian kernel.

    Parameters
    ----------
    image_data : 2D np.ndarray
        Image data (e.g. subbkg_img.data).
    kernel_fwhm_arcsec : float
        FWHM of the Gaussian kernel in arcsec.
    pixelscale : float
        Pixel scale in arcsec/pixel.
    """
    sigma2fwhm = 2 * np.sqrt(2 * np.log(2))
    sigma_pix = kernel_fwhm_arcsec / sigma2fwhm / pixelscale
    return gaussian_filter(image_data, sigma = sigma_pix)

# %%
# Cross-match m475 / m525 refcatalogs: analyze only common targets
from astropy.coordinates import SkyCoord

cat_m475 = target_img_m475.refcatalog.data
cat_m525 = target_img_m525.refcatalog.data
coords_m475 = SkyCoord(ra = cat_m475['X_WORLD'], dec = cat_m475['Y_WORLD'], unit = 'deg')
coords_m525 = SkyCoord(ra = cat_m525['X_WORLD'], dec = cat_m525['Y_WORLD'], unit = 'deg')
idx_m525, sep, _ = coords_m475.match_to_catalog_sky(coords_m525)
is_matched = sep.arcsec < 1.0
matched_cat_m475 = cat_m475[is_matched]
matched_cat_m525 = cat_m525[idx_m525[is_matched]]
mag_m475_matched = np.array(matched_cat_m475['MAGSKY_AUTO'])
mag_m525_matched = np.array(matched_cat_m525['MAGSKY_AUTO'])
print(f'matched {len(matched_cat_m475)} sources '
      f'(m475: {len(cat_m475)}, m525: {len(cat_m525)})')

# %%
# Measure Gaussian FWHM of both images (matched targets only)
pixscale_m475 = np.mean(subbkg_img_m475.pixelscale)
pixscale_m525 = np.mean(subbkg_img_m525.pixelscale)
fwhm_gauss_m475 = measure_gaussian_fwhm(
    subbkg_img_m475.data, matched_cat_m475['X_IMAGE'],
    matched_cat_m475['Y_IMAGE'], pixscale_m475, half_size = 15)
fwhm_gauss_m525 = measure_gaussian_fwhm(
    subbkg_img_m525.data, matched_cat_m525['X_IMAGE'],
    matched_cat_m525['Y_IMAGE'], pixscale_m525, half_size = 15)
median_fwhm_gauss_m475 = np.nanmedian(fwhm_gauss_m475)
median_fwhm_gauss_m525 = np.nanmedian(fwhm_gauss_m525)
print(f'm475 median Gaussian FWHM: {median_fwhm_gauss_m475:.2f} arcsec')
print(f'm525 median Gaussian FWHM: {median_fwhm_gauss_m525:.2f} arcsec')
std_fwhm_gauss_m475 = np.nanstd(fwhm_gauss_m475)
std_fwhm_gauss_m525 = np.nanstd(fwhm_gauss_m525)
print(f'm475 std Gaussian FWHM: {std_fwhm_gauss_m475:.2f} arcsec')
print(f'm525 std Gaussian FWHM: {std_fwhm_gauss_m525:.2f} arcsec')


# %%
# Convolve the smaller-seeing image to match the larger-seeing one
if median_fwhm_gauss_m475 < median_fwhm_gauss_m525:
    band, ref_band = 'm475', 'm525'
    subbkg_img_target = subbkg_img_m475
    catalog_target = matched_cat_m475
    mag_target = mag_m475_matched
    pixscale_target = pixscale_m475
    fwhm_gauss_before = fwhm_gauss_m475
    fwhm_gauss_ref = fwhm_gauss_m525
else:
    band, ref_band = 'm525', 'm475'
    subbkg_img_target = subbkg_img_m525
    catalog_target = matched_cat_m525
    mag_target = mag_m525_matched
    pixscale_target = pixscale_m525
    fwhm_gauss_before = fwhm_gauss_m525
    fwhm_gauss_ref = fwhm_gauss_m475

median_fwhm_before = np.nanmedian(fwhm_gauss_before)
median_fwhm_ref = np.nanmedian(fwhm_gauss_ref)
std_fwhm_before = np.nanstd(fwhm_gauss_before)
std_fwhm_ref = np.nanstd(fwhm_gauss_ref)
kernel_fwhm = np.sqrt(median_fwhm_ref**2 - median_fwhm_before**2)
print(f'convolving {band} ({median_fwhm_before:.2f} arcsec) '
      f'to match {ref_band} ({median_fwhm_ref:.2f} arcsec)')
print(f'kernel FWHM = {kernel_fwhm:.2f} arcsec')
conv_data = convolve_gaussian(subbkg_img_target.data, kernel_fwhm, pixscale_target)

# %%
# Re-measure Gaussian FWHM after convolution
fwhm_gauss_after = measure_gaussian_fwhm(
    conv_data, catalog_target['X_IMAGE'], catalog_target['Y_IMAGE'],
    pixscale_target, half_size = 25)
median_fwhm_after = np.nanmedian(fwhm_gauss_after)
std_fwhm_after = np.nanstd(fwhm_gauss_after)
print(f'[{band}] median FWHM before  : {median_fwhm_before:.2f} arcsec')
print(f'[{band}] median FWHM after   : {median_fwhm_after:.2f} arcsec')
print(f'[{ref_band}] target seeing     : {median_fwhm_ref:.2f} arcsec')

# %%
# Source shapes before / after convolution
show_source_cutouts(subbkg_img_target.data, catalog_target['X_IMAGE'],
                    catalog_target['Y_IMAGE'], mag_target,
                    title = f'{band} original', half_size = 10)
show_source_cutouts(conv_data, catalog_target['X_IMAGE'],
                    catalog_target['Y_IMAGE'], mag_target,
                    title = f'{band} convolved (kernel {kernel_fwhm:.2f} arcsec)',
                    half_size = 10)

# %%
# Residual between SExtractor FWHM and Gaussian FWHM (matched sources)
fwhm_sex_m475_matched = 3600 * np.array(matched_cat_m475['FWHM_WORLD'])
fwhm_sex_m525_matched = 3600 * np.array(matched_cat_m525['FWHM_WORLD'])
fig, axes = plt.subplots(2, 1, figsize = (10, 8))
axes[0].scatter(mag_m475_matched, fwhm_sex_m475_matched, edgecolor = 'r', facecolor = 'none',
            label = f'm475 (Source Extractor: {median_fwhm_m475_sex:.2f} ± {std_fwhm_m475_sex:.2f} arcsec)')
axes[0].scatter(mag_m475_matched, fwhm_gauss_m475, edgecolor = 'r', facecolor = 'r',
            label = f'm475 (Gaussian fit: {median_fwhm_gauss_m475:.2f} ± {std_fwhm_gauss_m475:.2f} arcsec)')
axes[0].axhline(median_fwhm_m475_sex, color = 'r', linestyle = ':')
axes[0].axhline(median_fwhm_gauss_m475, color = 'r', linestyle = '-')
axes[1].axhline(median_fwhm_m525_sex, color = 'b', linestyle = ':')
axes[1].axhline(median_fwhm_gauss_m525, color = 'b', linestyle = '--')
axes[1].scatter(mag_m525_matched, fwhm_sex_m525_matched, edgecolor = 'b', facecolor = 'none',
            label = f'm525 (Source Extractor: {median_fwhm_m525_sex:.2f} ± {std_fwhm_m525_sex:.2f} arcsec)')
axes[1].scatter(mag_m525_matched, fwhm_gauss_m525, edgecolor = 'b', facecolor = 'b',
            label = f'm525 (Gaussian fit: {median_fwhm_gauss_m525:.2f} ± {std_fwhm_gauss_m525:.2f} arcsec)')
axes[0].set_title('m475')
axes[0].set_ylabel('FWHM (arcsec)')
axes[0].legend(loc = 'upper right')

axes[1].set_title('m525')
axes[1].set_xlabel('mag_auto (AB, each band)')
axes[1].set_ylabel('FWHM (arcsec)')
axes[1].legend(loc = 'upper right')
plt.show()
# %%
# Per-source Gaussian FWHM change by convolution (arrows: before -> after)
valid = np.isfinite(fwhm_gauss_before) & np.isfinite(fwhm_gauss_after)
plt.figure(figsize = (8, 6))
plt.quiver(mag_target[valid], fwhm_gauss_before[valid],
           np.zeros(valid.sum()), (fwhm_gauss_after - fwhm_gauss_before)[valid],
           angles = 'xy', scale_units = 'xy', scale = 1,
           width = 0.002, color = 'gray', alpha = 0.6)
plt.scatter(mag_target[valid], fwhm_gauss_before[valid], s = 10, color = 'b',
            label = f'm475 ({median_fwhm_before:.2f} ± {std_fwhm_before:.2f} arcsec)')
plt.scatter(mag_target[valid], fwhm_gauss_after[valid], s = 10, color = 'r',
            label = f'm475 convolved ({median_fwhm_after:.2f} ± {std_fwhm_after:.2f} arcsec)')
plt.scatter(mag_target, fwhm_gauss_ref, s = 10, edgecolor = 'g', facecolor = 'none',
            label = f'{ref_band} reference ({median_fwhm_ref:.2f} ± {std_fwhm_ref:.2f} arcsec)')
xlim = plt.xlim()
plt.fill_between(xlim, median_fwhm_before - std_fwhm_before, median_fwhm_before + std_fwhm_before, color = 'b', alpha = 0.05)
plt.fill_between(xlim, median_fwhm_after - std_fwhm_after, median_fwhm_after + std_fwhm_after, color = 'r', alpha = 0.05)
plt.fill_between(xlim, median_fwhm_ref - std_fwhm_ref, median_fwhm_ref + std_fwhm_ref, color = 'g', alpha = 0.05)
plt.xlim(xlim)
plt.axhline(median_fwhm_ref, color = 'g', linestyle = '--')
plt.axhline(median_fwhm_before, color = 'b', linestyle = '--')
plt.axhline(median_fwhm_after, color = 'r', linestyle = '--')
plt.xlabel(f'mag_auto (AB, {band})')
plt.ylabel('FWHM (arcsec, from Gaussian fit)')
plt.title(f'PSF matching: {band} -> {ref_band}')
plt.legend()
plt.show()

# %%
