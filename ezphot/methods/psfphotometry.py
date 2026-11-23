#%%
import inspect
from typing import Union, Optional

import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, mad_std
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from photutils.aperture import CircularAnnulus, ApertureStats

from ezphot.methods import ErrormapGenerator, PhotometricCalibration
from ezphot.helper import Helper
from ezphot.imageobjects import (
    ScienceImage, ReferenceImage,
    Background, Mask, Errormap
)
from ezphot.dataobjects import Catalog
from ezphot.skycatalog import SkyCatalogUtility
#%%

# FLAG_PSF bits
FLAG_FALLBACK = 1   # the PSF at the position comes (partly) from a Moffat fallback tile of build_epsf (psf_info given)
FLAG_MASKED = 2     # masked or invalid pixels inside the fit radius (weight 0)
FLAG_EDGE = 4       # the fit radius crosses the image edge
FLAG_FAILED = 8     # no fit: position not finite or off the image, background not finite, too few usable pixels
FLAG_RECENTER = 16  # psf_photometry recenter: the position fit moved more than recenter_max_shift, or failed
#                     ('offset': the applied offset exceeds recenter_max_shift, an image-level warning, not a bad fit)
FLAG_NEIGHBOR_EXT = 32   # psf_photometry neighbours: an extended neighbour (neighbor_extended) nearby, not fitted

# build_epsf model='poly' with grid 1: evaluation tiles of about this size (pixels) along x and y
_POLY_TILE_PIXEL = 512
# build_epsf model='poly' order ladder: an order k >= 1 needs >= _POLY_ZONE_STARS usable stars in each of the
# N x N equal zones of the image, N = _POLY_ZONES[k] (orders above 3: _POLY_ZONES[3]), besides 5 x its coefficients
# stars
_POLY_ZONES = {1: 2, 2: 2, 3: 3}
_POLY_ZONE_STARS = 1
# build_epsf recenter_iter: a star whose fitted centre moves more than this (pixels) in one round keeps its centre
_EPSF_RECENTER_LIMIT = 1.0
# psf_photometry recenter='offset': a plane per axis from >= _OFFSET_MIN_PLANE measured sources, the median from
# >= _OFFSET_MIN_CONST, else no offset; one _OFFSET_CLIP-sigma (MAD) clipping of the plane, then the final fit
_OFFSET_MIN_PLANE = 10
_OFFSET_MIN_CONST = 3
_OFFSET_CLIP = 3.0
# psf_photometry neighbours: the neighbour_x/y sources within the fit-pixel radius + _NEIGHBOR_RADIUS_FWHM x FWHM of a
# source are fitted with it, except the nearest one within _NEIGHBOR_SELF_FWHM x FWHM (the source's own detection) and
# any closer than _NEIGHBOR_MIN_FWHM x FWHM (not separable: duplicate entries, deblending pieces)
_NEIGHBOR_RADIUS_FWHM = 2.0
_NEIGHBOR_SELF_FWHM = 1.0
_NEIGHBOR_MIN_FWHM = 0.5

# build_epsf star cuts on the sky-subtracted star box (FWHM = header seeing):
# second peak: a local maximum of the box smoothed with a Gaussian of the FWHM, 1.5 FWHM < r <= R + 11 from the star
# and off the box border, above max(_PEAK2_SNR x the smoothed noise, _PEAK2_FRACTION x the smoothed star peak)
# (a neighbour, galaxy or residue of >= 5 % of the star; the wing of a smooth PSF has no local maximum)
_PEAK2_SNR = 5.0
_PEAK2_FRACTION = 0.05
# spike: a pixel within R of the star above the mean a of its 8 neighbours by more than both _SPIKE_RATIO x a
# (above 2 a: a PSF pixel exceeds its neighbour mean only where the profile is concave, most at the centre, and the
# centre of a star with FWHM >~ 3 px is below 2 a, whatever the header seeing (at 2.5 px a few % of normal stars are
# dropped); a median of the neighbours would be pulled low on a steep wing) and _SPIKE_SNR x sqrt(noise^2 + a / EGAIN)
# (8 sigma keeps noise out in a box of a few thousand pixels): a cosmic ray or bad pixel; skipped without EGAIN (the
# star's Poisson noise of the noise condition is then undefined)
_SPIKE_RATIO = 1.0
_SPIKE_SNR = 8.0
_NEIGHBOUR_MEAN8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8


def _bad_pixels(a: np.ndarray) -> np.ndarray:
    """Not finite, or nonzero with |value| < 1e-20 (the 1e-30 hotpants writes for invalid pixels; ADU-scale values
    assumed). An exact 0 is valid data: raw - master bias - master dark of integer-valued masters on a low-sky frame
    (empty areas are NaN in stacks and, from 0.9.0, in reprojections incl. DIA sci_/ref_ images (SWarp weight 0 ->
    NaN), 1e-30 in hotpants difference images). Pass empty areas that are 0 (e.g. the no-weight SWarp edges of
    reprojections and DIA sci_ images written before 0.9.0 or with weight0_tonan=False) as target_mask."""
    return ~np.isfinite(a) | ((a != 0) & (a > -1e-20) & (a < 1e-20))


def _grid_xy(grid):
    """(nx, ny) tiles along x (columns) and y (rows) of grid: an int N (N x N) or (nx, ny)."""
    g = (grid, grid) if np.ndim(grid) == 0 else tuple(grid)
    if len(g) != 2 or any(int(v) != v or v < 1 for v in g):
        raise ValueError(f"grid must be a positive int N or a pair (nx, ny) of them, got {grid!r}")
    return int(g[0]), int(g[1])


def _tiles(shape, grid):
    """Equal tiling of an image into grid = N (N x N) or (nx, ny) tiles: tile centres xc (columns), yc (rows) and
    edges xe, ye (0-based pixel)."""
    ny, nx = shape
    gx, gy = _grid_xy(grid)
    xe = np.linspace(0, nx, gx + 1).round().astype(int)
    ye = np.linspace(0, ny, gy + 1).round().astype(int)
    return (xe[:-1] + xe[1:]) // 2, (ye[:-1] + ye[1:]) // 2, xe, ye


def _tent_weights(v: float, centres) -> np.ndarray:
    """Weights of the tiles along one axis at coordinate v: linear between neighbouring centres, 1 outside the outer
    centres (a partition of unity)."""
    n = len(centres)
    w = np.zeros(n)
    if n == 1 or v <= centres[0]:
        w[0] = 1.0
    elif v >= centres[-1]:
        w[-1] = 1.0
    else:
        k = int(np.searchsorted(centres, v, side='right')) - 1
        f = (v - centres[k]) / (centres[k + 1] - centres[k])
        w[k], w[k + 1] = 1.0 - f, f
    return w


def _centroid(d: np.ndarray, x0: float, y0: float, sigma: float, n: int = 10):
    """Gaussian-weighted (sigma, pixels) iterative centroid of the positive part of d, starting at (x0, y0)."""
    yy, xx = np.indices(d.shape)
    x, y = x0, y0
    for _ in range(n):
        w = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)) * np.clip(d, 0, None)
        s = w.sum()
        if s <= 0:
            break
        x, y = (w * xx).sum() / s, (w * yy).sum() / s
    return x, y


def _fit_shift(d: np.ndarray, p: np.ndarray, w: np.ndarray, dx: float, dy: float, bkg: np.ndarray = None,
               limit: float = 1.0, max_iter: int = 20, tol: float = 1e-3, h: float = 0.05):
    """Position of a point source by Gauss-Newton: d = F P(dx, dy) (+ bkg @ c with bkg), weighted least squares over
    the pixels with weight w > 0. d, w are (m, m), p is (n, n) with n <= m (zero-padded to m after the shift); bkg,
    optional, (n_w, k) holds k background columns over the n_w pixels w > 0 in row-major order (e.g. 1, or 1, x, y).
    P(dx, dy) is p shifted by (dy, dx)
    (scipy.ndimage.shift, order 3) and normalised to sum 1, as psf_photometry places the PSF, so (dx, dy) is the source
    offset from the stamp centre m // 2. Starts at (dx, dy) with the linear F (and c); position derivatives by central
    differences of +-h px. Returns the fitted (dx, dy), or None: singular system, not finite, F <= 0, a move of more
    than limit px from the start, or no step below tol px in max_iter steps."""
    pad = (d.shape[0] - p.shape[0]) // 2

    def model(sx, sy):
        m = ndimage.shift(p, (sy, sx), order=3, mode='constant')
        m = m / m.sum()
        return np.pad(m, pad) if pad else m

    g = w > 0
    W, D = w[g], d[g]
    x0, y0 = dx, dy
    try:
        P = model(dx, dy)[g]
        X = np.column_stack([P, bkg]) if bkg is not None else P[:, None]
        lin = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * D))
        F, c = lin[0], lin[1:]
        for _ in range(max_iter):
            P = model(dx, dy)[g]
            Px = (model(dx + h, dy)[g] - model(dx - h, dy)[g]) / (2 * h)
            Py = (model(dx, dy + h)[g] - model(dx, dy - h)[g]) / (2 * h)
            J = np.column_stack([P, F * Px, F * Py] + ([bkg] if bkg is not None else []))
            C = bkg @ c if bkg is not None else 0.0
            step = np.linalg.solve(J.T @ (W[:, None] * J), J.T @ (W * (D - F * P - C)))
            F, dx, dy = F + step[0], dx + step[1], dy + step[2]
            if bkg is not None:
                c = c + step[3:]
            if not (np.all(np.isfinite(step)) and F > 0 and np.hypot(dx - x0, dy - y0) <= limit):
                return None
            if np.hypot(step[1], step[2]) < tol:
                return dx, dy
    except np.linalg.LinAlgError:
        return None
    return None


def _offset_model(x, y, dx, dy, shape):
    """Smooth position offset over the image from offsets (dx, dy) measured at (x, y) (NaN: not measured), per
    axis a + b u + c v (u, v: x, y scaled from the first/last pixel to -1/+1): with >= _OFFSET_MIN_PLANE measured a
    least-squares plane of all, one clipping of the sources beyond _OFFSET_CLIP sigma from the median residual on
    either axis (sigma = 1.4826 MAD of the residuals per axis, at least 1e-3 px) and the final plane of the kept ones
    (all kept when fewer than _OFFSET_MIN_PLANE would remain); with >= _OFFSET_MIN_CONST the median (b = c = 0, no
    clipping); else none (all 0). Returns (a, b, c) for dx and for dy, the sources used, the sources measured and the
    residual rms (x, y) of the used ones about the returned model (NaN without a model)."""
    ny, nx = shape
    ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(x) & np.isfinite(y)
    n_meas = int(ok.sum())
    if n_meas < _OFFSET_MIN_CONST:
        return np.zeros(3), np.zeros(3), 0, n_meas, (np.nan, np.nan)
    A = np.column_stack([np.ones(n_meas), 2 * x[ok] / (nx - 1) - 1, 2 * y[ok] / (ny - 1) - 1])
    ddx, ddy = dx[ok], dy[ok]
    keep = np.ones(n_meas, dtype=bool)
    if n_meas < _OFFSET_MIN_PLANE:
        cx, cy = np.array([np.median(ddx), 0.0, 0.0]), np.array([np.median(ddy), 0.0, 0.0])
    else:
        rx = ddx - A @ np.linalg.lstsq(A, ddx, rcond=None)[0]
        ry = ddy - A @ np.linalg.lstsq(A, ddy, rcond=None)[0]
        clip = ((np.abs(rx - np.median(rx)) <= _OFFSET_CLIP * max(mad_std(rx), 1e-3))
                & (np.abs(ry - np.median(ry)) <= _OFFSET_CLIP * max(mad_std(ry), 1e-3)))
        if clip.sum() >= _OFFSET_MIN_PLANE:
            keep = clip
        cx = np.linalg.lstsq(A[keep], ddx[keep], rcond=None)[0]   # the final plane of the kept sources
        cy = np.linalg.lstsq(A[keep], ddy[keep], rcond=None)[0]
    rx, ry = ddx - A @ cx, ddy - A @ cy
    rms = (float(np.sqrt(np.mean(rx[keep] ** 2))), float(np.sqrt(np.mean(ry[keep] ** 2))))
    return cx, cy, int(keep.sum()), n_meas, rms


def _check_psf(a: np.ndarray, name: str):
    if a.ndim not in (2, 4) or a.shape[-1] != a.shape[-2] or a.shape[-1] % 2 == 0:
        raise ValueError(f"{name} must be an odd square array (n, n) or tiles of them (ny, nx, n, n), got shape "
                         f"{a.shape}")


class PSFPhotometry:
    """
    Method class for forced PSF photometry.

    This class provides methods

    1. Empirical PSF (ePSF) from isolated, unsaturated catalog (Gaia) stars of the image, optionally per tile of an
       nx x ny tiling or as a polynomial in the position evaluated at the tile centres, with a Moffat fallback
       (build_epsf, moffat_psf).

    2. Effective PSF of a filtered image: the PSF convolved with the filter kernel (effective_psf).

    3. Forced PSF photometry at fixed positions: linear least-squares fit of the PSF amplitude (psf_photometry),
       optionally together with neighbours at given fixed positions, catalog <img>.fits.psf.cat.

    4. PSF zero point ZP_PSF from the reference stars of the image (calibrate_zp).

    PSF arrays are odd square (2R+1, 2R+1) with the centre at index R, or (ny, nx, 2R+1, 2R+1) for the equal
    nx x ny tiling of the measured image (nx tiles along x, ny along y; tile [j, i] = row j, column i), blended at a
    position with tent weights between the tile centres. Pixel coordinates are 0-based (x = column, y = row).
    """
    def __init__(self):
        """
        Initialize the PSFPhotometry class.

        Parameters
        ----------
        None
        """
        self.helper = Helper()
        self.errormap = ErrormapGenerator()
        self.catalogutils = SkyCatalogUtility()

    def __repr__(self):
        return f"Method class: {self.__class__.__name__}\n For help, use 'help(self)' or `self.help()`."

    def help(self):
        # Get all public methods from the class, excluding `help`
        methods = [
            (name, obj)
            for name, obj in inspect.getmembers(self.__class__, inspect.isfunction)
            if not name.startswith("_") and name != "help"
        ]

        # Build plain text list with parameters
        lines = []
        for name, func in methods:
            sig = inspect.signature(func)
            params = [str(p) for p in sig.parameters.values() if p.name != "self"]
            sig_str = f"({', '.join(params)})" if params else "()"
            lines.append(f"- {name}{sig_str}")

        # Final plain text output
        help_text = ""
        self.helper.print(f"Help for {self.__class__.__name__}\n{help_text}\nPublic methods:\n" + "\n".join(lines), True)

    @staticmethod
    def _fwhm_pixel(target_img):
        """(FWHM in pixels from the header seeing, pixel scale in arcsec/pixel)."""
        scale = target_img.pixelscale
        if scale is None:
            raise ValueError(f"{target_img.path.name}: no celestial WCS (pixel scale)")
        scale = float(np.mean(scale))
        if target_img.seeing is None:
            raise ValueError(f"{target_img.path.name}: no seeing (header SEEING) for the PSF size")
        return float(target_img.seeing) / scale, scale

    def _gaia_isolated(self, target_img, catalog_type, catalog_version, g_lower, g_upper, isolation_radius_arcsec,
                       verbose):
        """0-based pixel positions of the catalog stars with g_lower < g_mean < g_upper and no other catalog star
        (any magnitude) within isolation_radius_arcsec."""
        catalogs = self.catalogutils.get_catalogs(
            ra = target_img.ra,
            dec = target_img.dec,
            fov_ra = target_img.fovx,
            fov_dec = target_img.fovy,
            objname = target_img.objname,
            catalog_type = catalog_type,
            catalog_version = catalog_version,
            verbose = verbose
        )
        if len(catalogs) == 0:
            raise ValueError("No catalogs found in the given coordinates.")
        stars = vstack([catalog.data for catalog in catalogs])
        if 'id' in stars.colnames:   # overlapping catalog tiles: a star listed twice would not be isolated
            stars = unique(stars, keys='id')
        ra = np.ma.filled(np.ma.asarray(stars['ra'], dtype=float), np.nan)
        dec = np.ma.filled(np.ma.asarray(stars['dec'], dtype=float), np.nan)
        g = np.ma.filled(np.ma.asarray(stars['g_mean'], dtype=float), np.nan)
        finite = np.isfinite(ra) & np.isfinite(dec)
        ra, dec, g = ra[finite], dec[finite], g[finite]
        sel = (g > g_lower) & (g < g_upper)
        if sel.sum() == 0 or len(ra) < 2:
            return np.array([]), np.array([])
        coord = SkyCoord(ra, dec, unit='deg')
        _, d2, _ = coord[sel].match_to_catalog_sky(coord, nthneighbor=2)
        iso = d2.arcsec > isolation_radius_arcsec
        x, y = target_img.wcs.all_world2pix(ra[sel][iso], dec[sel][iso], 0)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    @staticmethod
    def moffat_psf(fwhm_pixel: float, radius_pixel: int, beta: float = 3.0, oversample: int = 5) -> np.ndarray:
        """
        Circular Moffat PSF, pixel-integrated (oversample x oversample sub-pixels), zero beyond radius_pixel, sum 1.

        Parameters
        ----------
        fwhm_pixel : float
            FWHM in pixels.
        radius_pixel : int
            R: the array is (2R+1, 2R+1) with the centre at index R.
        beta : float, optional
            Moffat beta. Defaults to 3.
        oversample : int, optional
            Sub-pixels per pixel and axis. Defaults to 5.

        Returns
        -------
        psf : np.ndarray
        """
        R = int(radius_pixel)
        alpha = fwhm_pixel / (2 * np.sqrt(2 ** (1 / beta) - 1))
        n = 2 * R + 1
        t = (np.arange(n * oversample) + 0.5) / oversample - 0.5 - R
        xx, yy = np.meshgrid(t, t)
        m = (1 + (xx ** 2 + yy ** 2) / alpha ** 2) ** (-beta)
        m = m.reshape(n, oversample, n, oversample).mean(axis=(1, 3))
        yy, xx = np.indices(m.shape)
        m[np.hypot(xx - R, yy - R) > R] = 0
        return m / m.sum()

    @staticmethod
    def effective_psf(psf: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """
        Effective PSF of an image filtered with the kernel psi (e.g. a difference image after a decorrelation
        filter): the full linear convolution psf (x) psi (scipy.signal.fftconvolve, mode 'full').

        Parameters
        ----------
        psf : np.ndarray
            PSF of the unfiltered image: (n, n) or (ny, nx, n, n), n odd.
        psi : np.ndarray
            Filter kernel: (m, m) or (ny, nx, m, m), m odd, centre at index m // 2.

        Returns
        -------
        psf_eff : np.ndarray
            (n + m - 1, n + m - 1) when both are 2-D, else (ny, nx, n + m - 1, n + m - 1): tile [j, i] is the
            PSF tile convolved with the psi tile (a 2-D input is used for every tile; two tiled inputs need the same
            (ny, nx)). Not renormalised: the sum is sum(psf) x sum(psi) (psi(0) = 1 keeps it; psf_photometry
            normalises to 1 anyway). A tiled psi must come from the equal nx x ny tiling of the image that
            psf_photometry uses (same tile centres, blended with the same tent weights). Tile by tile is an
            approximation: blending the products of two tiled inputs differs from the product of the blends by
            second-order cross terms between neighbouring tiles. A tiled psi is one array: tile kernels of different
            sizes are zero-padded (centred) to a common odd size m by the caller.
        """
        psf = np.asarray(psf, dtype=float)
        psi = np.asarray(psi, dtype=float)
        _check_psf(psf, 'psf')
        _check_psf(psi, 'psi')
        if psf.ndim == 2 and psi.ndim == 2:
            return signal.fftconvolve(psf, psi, mode='full')
        if psf.ndim == 4 and psi.ndim == 4 and psf.shape[:2] != psi.shape[:2]:
            raise ValueError(f"psf and psi tiles differ: (ny, nx) {psf.shape[:2]} and {psi.shape[:2]}")
        gy, gx = psf.shape[:2] if psf.ndim == 4 else psi.shape[:2]
        return np.array([[signal.fftconvolve(psf[j, i] if psf.ndim == 4 else psf,
                                             psi[j, i] if psi.ndim == 4 else psi, mode='full')
                          for i in range(gx)] for j in range(gy)])

    def build_epsf(self,
                   target_img: Union[ScienceImage, ReferenceImage],
                   target_mask: Optional[Mask] = None,
                   star_x: Union[list, np.ndarray] = None,
                   star_y: Union[list, np.ndarray] = None,
                   catalog_type: str = 'GAIAXP',
                   catalog_version: str = 'v1',
                   g_lower: float = 12.5,
                   g_upper: float = 15.5,
                   isolation_radius_arcsec: float = 15.0,
                   peak_fraction: float = 0.5,
                   snr_lower: float = 50.0,
                   radius_pixel: int = None,
                   grid: Union[int, tuple] = 1,
                   model: str = 'grid',
                   poly_order: int = 3,
                   min_stars: int = 20,
                   max_stars: int = 300,
                   max_stars_total: int = 1000,
                   beta: float = 3.0,
                   seed: int = 0,
                   verbose: bool = True,
                   recenter_iter: int = 0,
                   snr_lower_fallback: Optional[float] = 30.0):
        """
        Empirical PSF of the image: median of normalised, sub-pixel re-centred cut-outs of isolated unsaturated stars.

        Stars: catalog stars (catalog_type, archived or queried as in photometric_calibration) with
        g_lower < g_mean < g_upper and no other catalog star within isolation_radius_arcsec, or the given positions.
        Per star (box half-width R + 11 px inside the image, the whole box free of invalid pixels, not finite or
        nonzero with |value| < 1e-20 (an exact 0 is valid), and of target_mask): minus the median of the ring
        R + 4 < r <= R + 11 px; kept if the net peak
        (central 5 x 5) < peak_fraction x SATURATE - that sky (header SATURATE, 5e4 without it) and > snr_lower x the
        MAD std of the pixels beyond R + 4 px, and dropped when the box has a second peak or a spike: a local maximum of
        the box smoothed with a Gaussian of the FWHM at 1.5 FWHM < r <= R + 11 px (not on the box border) above
        max(5 x the MAD std of the smoothed box beyond R + 4 px, 0.05 x the smoothed peak within 1.5 FWHM) (a
        neighbour star, a galaxy, a residue), or a pixel at r <= R above the mean a of its 8 neighbours by more than
        both a (twice the mean; a star centre with FWHM >~ 3 px is below it; at 2.5 px a few % of normal stars, and
        more of more undersampled ones, may be dropped) and 8 x sqrt(noise^2 + a / EGAIN) (a cosmic ray or bad pixel;
        without a positive finite EGAIN the star's Poisson noise in that condition is undefined and this spike cut is
        skipped conservatively, with a warning);
        centroid (Gaussian-weighted, sigma = FWHM / 2.355), cubic-spline shift
        (scipy.ndimage.shift, order 3) to the box centre, cut to (2R+1, 2R+1), normalised to sum 1. The FWHM is the
        header seeing. The image and its files are only read.

        model 'grid' (default): per tile, up to max_stars stars of the tile in a random order (fixed seed); the
        median of >= min_stars cut-outs, with negative wings beyond 2 FWHM and everything beyond R set to 0,
        normalised to sum 1; with fewer, the tile is moffat_psf(FWHM, R, beta) and a warning is printed.
        model 'poly' (PSFEx/Piff-like pixel basis): up to max_stars_total stars of the whole image in a random order
        (fixed seed); every pixel of the cut-outs is a polynomial of total order poly_order in the star position
        (x^a y^b, a + b <= poly_order, x and y scaled from the image's first/last pixel to -1/+1), fitted by linear
        least squares (a mean, not a median) to all of these stars (no outlier rejection). The first fit is
        unweighted; its model M gives the per-star variance of the normalised cut-out, var = (noise / S)^2 +
        max(M, 0) / (EGAIN x S) (noise = the MAD std above, S = the cut-out sum; without a positive finite EGAIN the
        Poisson term is left out, with a warning), and the fit is repeated once with one weight per star, 1 / the
        mean of var over r <= R. The polynomial is evaluated at the tile centres and post-processed
        as a median tile above. Positions beyond the outermost fitted stars are extrapolated (no flag).
        Order ladder: poly_order is the highest order; an order k >= 1 is used when there are >= 5 x its
        (k + 1)(k + 2) / 2 coefficients usable stars (order 3: 50, 2: 30, 1: 15; min_stars does not enter) and >= 1
        usable star in each equal zone of the image: 3 x 3 zones (x and y in thirds) for order 3 (and higher),
        2 x 2 (halves) for orders 2 and 1; else the next lower order is tried (order 0 only when poly_order is 0:
        >= 5 stars, no zones). No order fits: model 'grid' with grid 1 from the same stars (one median ePSF, 2-D;
        the Moffat fallback below min_stars), info 'model' 'grid'. A lower order or the median ePSF is announced by
        one warning with the reason of the order poly_order (star count, empty zones).

        S/N fallback: when the stars passing the cuts at snr_lower are fewer than the stars the order poly_order needs
        (model 'poly') or than min_stars in a tile (model 'grid'), the selection of those stars is repeated (same order
        of the stars) with the peak S/N limit snr_lower_fallback (if lower than snr_lower; None: off), with a warning;
        the stars of the repeat are used (model 'poly': for the order ladder above) whatever their number.

        recenter_iter > 0 (default 0: the centroid alone): after the PSF above, recenter_iter rounds of: the PSF
        (model 'grid': the tile's PSF; 'poly': the polynomial at the star, post-processed as a tile) shifted (cubic
        spline) to put its first moment at the array centre; per star a Gauss-Newton fit of position and flux of that
        PSF to the sky-subtracted box over r <= R (uniform weights); the cut-out re-made at the fitted centre; the PSF
        re-made from the same stars (same cuts, judged once; 'poly': the polynomial at the catalog positions) and
        shifted to put its first moment at the array centre. A star whose fit fails or moves its centre by more than
        1 px in a round keeps its centre of the previous round. Moffat tiles are not re-made.

        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The image (header SEEING and a celestial WCS needed).
        target_mask : Mask, optional
            Pixels (nonzero) kept out of the star boxes (an exact 0 is valid data: pass 0-filled empty areas,
            e.g. the SWarp no-weight edges of DIA sci_ images written before 0.9.0, here; from 0.9.0 the
            reprojection sets them to NaN).
        star_x, star_y : array-like, optional
            0-based pixel positions of the stars to use instead of the catalog selection (no isolation check).
        catalog_type, catalog_version : str, optional
            Star catalog. Defaults to 'GAIAXP', 'v1'.
        g_lower, g_upper : float, optional
            Gaia G range (g_mean). Defaults to 12.5, 15.5.
        isolation_radius_arcsec : float, optional
            No other catalog star within this radius. Defaults to 15.
        peak_fraction : float, optional
            Saturation cut as a fraction of SATURATE. Defaults to 0.5.
        snr_lower : float, optional
            Net peak / noise lower limit. Defaults to 50.
        radius_pixel : int, optional
            R. Defaults to ceil(max(3 FWHM, 8 arcsec)).
        grid : int or tuple, optional
            Tiling of the image: N for N x N tiles or (nx, ny) for nx tiles along x (columns) and ny along y (rows).
            Model 'grid': 1 (default) is one PSF for the image, else one PSF per tile (stars inside the tile).
            Model 'poly': the tiles whose centres the polynomial is evaluated at; 1 or (1, 1) is an automatic tiling
            of about 512 x 512 px tiles (round(image size / 512) per axis, at least 1).
        model : str, optional
            'grid' (default) or 'poly' (see above).
        poly_order : int, optional
            Model 'poly': highest total order of the polynomial in x, y (the order ladder above). Defaults to 3.
        min_stars : int, optional
            Fewer usable stars in a tile (model 'grid', or the median ePSF of model 'poly' when no order fits):
            Moffat fallback. Not used by the polynomial orders (order ladder above). Defaults to 20.
        max_stars : int, optional
            Model 'grid': at most this many stars per tile. Defaults to 300.
        max_stars_total : int, optional
            Model 'poly': at most this many stars in the image. Defaults to 1000.
        beta : float, optional
            Moffat beta of the fallback. Defaults to 3.
        seed : int, optional
            Random order of the stars. Defaults to 0.
        verbose : bool, optional
            Whether to print the star counts.
        recenter_iter : int, optional
            Rounds of PSF-fitted star centres (see above). Defaults to 0 (Gaussian-weighted centroid only).
        snr_lower_fallback : float, optional
            Peak S/N limit of the S/N fallback above. Defaults to 30; None: no fallback.

        Returns
        -------
        psf : np.ndarray
            (2R+1, 2R+1) for a single tile (grid 1 or (1, 1), or model 'poly' when no order fits), else
            (ny, nx, 2R+1, 2R+1); each PSF sums to 1.
        info : dict
            'source' (ny, nx) array of 'epsf' or 'moffat', 'nstars' (ny, nx) usable stars per tile (model 'poly':
            'epsf' and the stars of the fit in every tile), 'fwhm_pixel', 'radius_pixel', 'grid' (an int grid as
            given when that tiling was used, else the (nx, ny) used), 'model' (the model used: 'grid' for the
            no-order fallback of 'poly'), 'poly_order' (the order used; None for model 'grid'), 'psf_model' ('poly0'
            ... 'poly3' ... (the order used), 'grid' (median ePSF, at least one tile 'epsf') or 'moffat' (every tile
            Moffat)), 'recenter_iter', 'recenter' (one dict per round: 'n_fit' stars with a new centre, 'n_kept'
            stars that kept theirs, 'median_shift' and 'max_shift' of the new centres in px, NaN without one),
            'snr_lower_used' (ny, nx) peak S/N limit of the stars of each tile (snr_lower, or snr_lower_fallback after
            the S/N fallback; model 'poly': one value in every tile).
        """
        if model not in ('grid', 'poly'):
            raise ValueError(f"model must be 'grid' or 'poly', got {model!r}")
        if model == 'poly' and (not isinstance(poly_order, (int, np.integer)) or poly_order < 0):
            raise ValueError(f"poly_order must be an int >= 0, got {poly_order!r}")
        if not isinstance(recenter_iter, (int, np.integer)) or recenter_iter < 0:
            raise ValueError(f"recenter_iter must be an int >= 0, got {recenter_iter!r}")
        gx, gy = _grid_xy(grid)
        fwhm, scale = self._fwhm_pixel(target_img)
        R = int(radius_pixel) if radius_pixel else int(np.ceil(max(3 * fwhm, 8.0 / scale)))
        data = self.helper.to_native(target_img.data)
        ny, nx = data.shape
        half = R + 11   # star box half-width: sky ring R + 4 < r <= R + 11
        edge = R + 12
        if star_x is None:
            x, y = self._gaia_isolated(target_img, catalog_type, catalog_version, g_lower, g_upper,
                                       isolation_radius_arcsec, verbose)
        else:
            x = np.atleast_1d(np.asarray(star_x, dtype=float))
            y = np.atleast_1d(np.asarray(star_y, dtype=float))
        inside = np.isfinite(x) & np.isfinite(y) & (x >= edge) & (x < nx - edge) & (y >= edge) & (y < ny - edge)
        x, y = x[inside], y[inside]
        mask = None
        if target_mask is not None:
            mask = np.asarray(self.helper.to_native(target_mask.data)).astype(bool)
        try:
            saturation = float(target_img.header['SATURATE'])
        except (KeyError, TypeError, ValueError):
            saturation = 5e4
            print(f"[WARNING] {target_img.path.name}: no SATURATE in the header; PSF star peak cut at "
                  f"{peak_fraction} x 5e4 ADU")

        if model == 'poly' and (gx, gy) == (1, 1):   # automatic evaluation tiles of about _POLY_TILE_PIXEL px
            gx, gy = max(1, int(round(nx / _POLY_TILE_PIXEL))), max(1, int(round(ny / _POLY_TILE_PIXEL)))
        xc, yc, xe, ye = _tiles(data.shape, (gx, gy))
        rng = np.random.default_rng(seed)
        n = 2 * R + 1
        psf = np.zeros((gy, gx, n, n))
        source = np.full((gy, gx), 'epsf', dtype='<U6')
        nstars = np.zeros((gy, gx), dtype=int)
        yy, xx = np.indices((2 * half + 1, 2 * half + 1))
        rr_psf = np.hypot(*(np.indices((n, n)) - R))
        try:
            gain = float(target_img.egain)
        except (TypeError, ValueError):
            gain = np.nan
        if not (np.isfinite(gain) and gain > 0):
            gain = None
            print(f"[WARNING] {target_img.path.name}: no positive EGAIN; the spike cut of the PSF star selection is "
                  f"skipped")

        def cut_out(k, snr):
            """The stamp() tuple of star k (sky-subtracted, re-centred cut-out normalised to sum 1, ...), or None when
            it fails the cuts (peak S/N limit snr)."""
            xa, ya = int(round(x[k])), int(round(y[k]))
            box = (slice(ya - half, ya + half + 1), slice(xa - half, xa + half + 1))
            d = np.asarray(data[box], dtype=float)
            if _bad_pixels(d).any() or (mask is not None and mask[box].any()):
                return None
            rr = np.hypot(xx - (x[k] - xa + half), yy - (y[k] - ya + half))
            sky = np.median(d[(rr > R + 4) & (rr <= R + 11)])
            d = d - sky
            noise = mad_std(d[rr > R + 4])
            peak = d[half - 2:half + 3, half - 2:half + 3].max()
            if not (peak < peak_fraction * saturation - sky and peak > snr * noise):
                return None
            # spike (cosmic ray, bad pixel): above twice its 8-neighbour mean and far above the noise; only with
            # EGAIN (without it the star's Poisson noise in the noise condition is undefined, so skipped conservatively)
            if gain is not None:
                nb_mean = ndimage.convolve(d, _NEIGHBOUR_MEAN8, mode='nearest')
                nb_pos = np.clip(nb_mean, 0, None)
                excess = d - nb_mean
                if ((rr <= R) & (excess > _SPIKE_RATIO * nb_pos)
                        & (excess > _SPIKE_SNR * np.sqrt(noise ** 2 + nb_pos / gain))).any():
                    return None
            # second peak (neighbour, galaxy, residue): a local maximum of the smoothed box away from the star
            ds = ndimage.gaussian_filter(d, fwhm / 2.355)
            local_max = ds == ndimage.maximum_filter(ds, size=3)
            local_max[[0, -1], :] = False   # box border: the rising wing of a source outside the box
            local_max[:, [0, -1]] = False
            level = max(_PEAK2_SNR * mad_std(ds[rr > R + 4]), _PEAK2_FRACTION * ds[rr <= 1.5 * fwhm].max())
            if (local_max & (rr > 1.5 * fwhm) & (rr <= R + 11) & (ds > level)).any():
                return None
            cx, cy = _centroid(d, x[k] - xa + half, y[k] - ya + half, fwhm / 2.355)
            return stamp(d, cx, cy, noise)

        def stamp(d, cx, cy, noise):
            """(Cut-out of the sky-subtracted box d shifted to put (cx, cy) at the box centre, normalised to sum 1,
            noise / sum, the sum S, then d (None with recenter_iter 0), cx, cy, noise for recenter_iter), or None when
            S <= 0."""
            shifted = ndimage.shift(d, (half - cy, half - cx), order=3, mode='constant')
            cut = shifted[half - R:half + R + 1, half - R:half + R + 1]
            s = cut.sum()
            return (cut / s, noise / s, s, d if recenter_iter else None, cx, cy, noise) if s > 0 else None

        def finish(p):
            """Negative wings beyond 2 FWHM and everything beyond R set to 0, normalised to sum 1."""
            p[(rr_psf > 2 * fwhm) & (p < 0)] = 0
            p[rr_psf > R] = 0
            return p / p.sum()

        def anchor(p):
            """p shifted (cubic spline) to put its first moment at the array centre, then finish()."""
            my, mx = (np.indices((n, n)) * p).sum(axis=(1, 2)) / p.sum()
            return finish(ndimage.shift(p, (R - my, R - mx), order=3, mode='constant'))

        yn, xn = np.indices((n, n))

        def recentre(c, ref):
            """(Cut-out tuple c re-made at the star centre fitted with the PSF ref (_fit_shift over r <= R of the
            current centre, uniform weights), the move in px), or (c, NaN) when the fit fails or moves more than
            _EPSF_RECENTER_LIMIT px."""
            box, cx, cy, noise = c[3:]
            ix, iy = int(round(cx)), int(round(cy))
            if ix < R or iy < R or ix + R >= box.shape[1] or iy + R >= box.shape[0]:
                return c, np.nan
            dx, dy = cx - ix, cy - iy
            w = (np.hypot(xn - R - dx, yn - R - dy) <= R).astype(float)
            fit = _fit_shift(box[iy - R:iy + R + 1, ix - R:ix + R + 1], ref, w, dx, dy, limit=_EPSF_RECENTER_LIMIT)
            new = None if fit is None else stamp(box, ix + fit[0], iy + fit[1], noise)
            if new is None:
                return c, np.nan
            return new, float(np.hypot(new[4] - cx, new[5] - cy))

        def median_tile(cuts, where):
            """(Median ePSF of the cut-outs, 'epsf'), or (the Moffat fallback, 'moffat') below min_stars (warning)."""
            if len(cuts) >= min_stars:
                return finish(np.median(np.array(cuts), axis=0)), 'epsf'
            print(f"[WARNING] {target_img.path.name}: {len(cuts)} PSF stars (< {min_stars}) in {where}; "
                  f"Moffat beta={beta} PSF of the header seeing used")
            return self.moffat_psf(fwhm, R, beta), 'moffat'

        def collect(idx, snr, limit):
            """Cut-out tuples and positions of the stars idx (in that order) passing the cuts at peak S/N limit snr, at
            most limit stars."""
            cuts, xs, ys = [], [], []   # cuts: (cut-out, noise / S, S, box, cx, cy, noise)
            for k in idx:
                cut = cut_out(k, snr)
                if cut is not None:
                    cuts.append(cut)
                    xs.append(x[k])
                    ys.append(y[k])
                    if len(cuts) >= limit:
                        break
            return cuts, xs, ys

        def collect_fallback(idx, limit, enough, where):
            """collect() at snr_lower, repeated at snr_lower_fallback (warning) with fewer than enough stars; also the
            peak S/N limit used (no repeat when every star of idx already passed)."""
            cuts, xs, ys = collect(idx, snr_lower, limit)
            if (len(cuts) >= enough or len(cuts) == len(idx) or snr_lower_fallback is None
                    or not snr_lower_fallback < snr_lower):
                return cuts, xs, ys, float(snr_lower)
            n_first = len(cuts)
            cuts, xs, ys = collect(idx, snr_lower_fallback, limit)
            print(f"[WARNING] {target_img.path.name}: {n_first} PSF stars at peak S/N > {snr_lower:g} (< {enough}) in "
                  f"{where}; selection repeated at peak S/N > {snr_lower_fallback:g}: {len(cuts)} stars")
            return cuts, xs, ys, float(snr_lower_fallback)

        def need(order):
            """Usable stars for a polynomial of this order: 5 x its coefficients."""
            return 5 * (order + 1) * (order + 2) // 2

        tile_cuts = {}   # model 'grid': (j, i) -> the cut-out tuples of the tile, for recenter_iter
        snr_used = np.full((gy, gx), float(snr_lower))
        if model == 'grid':
            for j in range(gy):
                for i in range(gx):
                    in_tile = np.where((x >= xe[i]) & (x < xe[i + 1]) & (y >= ye[j]) & (y < ye[j + 1]))[0]
                    cuts, _, _, snr_used[j, i] = collect_fallback(rng.permutation(in_tile), max_stars, min_stars,
                                                                  f'tile ({j}, {i})')
                    nstars[j, i] = len(cuts)
                    psf[j, i], source[j, i] = median_tile([c[0] for c in cuts], f'tile ({j}, {i})')
                    tile_cuts[j, i] = cuts
        else:
            cuts, xs, ys, snr_used[:] = collect_fallback(rng.permutation(len(x)), max_stars_total, need(poly_order),
                                                         'the image')

            def zones(order):
                """N of the N x N zones of an order >= 1."""
                return _POLY_ZONES.get(order, _POLY_ZONES[3])

            def n_empty(order):
                """Zones of an order >= 1 with fewer than _POLY_ZONE_STARS stars."""
                if not xs:   # no usable star: every zone is empty
                    return zones(order) ** 2
                _, _, zxe, zye = _tiles(data.shape, zones(order))
                return int((np.histogram2d(ys, xs, bins=(zye, zxe))[0] < _POLY_ZONE_STARS).sum())

            # order ladder: poly_order down to 1 (0 only when asked); an order needs need(order) stars and, from
            # order 1, >= _POLY_ZONE_STARS stars in each of its zones
            order = next((k for k in (range(poly_order, 0, -1) if poly_order > 0 else [0])
                          if len(cuts) >= need(k) and (k == 0 or n_empty(k) == 0)), None)
            if order != poly_order:
                why = [f"{len(cuts)} PSF stars (< {need(poly_order)})"] if len(cuts) < need(poly_order) else []
                if poly_order > 0 and n_empty(poly_order):
                    why.append(f"{n_empty(poly_order)} of {zones(poly_order) ** 2} zones ({zones(poly_order)} x "
                               f"{zones(poly_order)}) without a PSF star")
                print(f"[WARNING] {target_img.path.name}: polynomial PSF of order {poly_order} not usable "
                      f"({'; '.join(why)}); " + (f"order {order} used" if order is not None else
                                                 "one median ePSF of them used (model 'grid', grid 1)"))
            if order is not None:
                # terms x^a y^b with a + b <= order
                powers = [(t - b, b) for t in range(int(order) + 1) for b in range(t + 1)]

                def basis(px, py):
                    """Polynomial terms at 0-based pixel positions, x and y scaled to [-1, 1] over the image."""
                    u = 2 * np.asarray(px, dtype=float) / (nx - 1) - 1
                    v = 2 * np.asarray(py, dtype=float) / (ny - 1) - 1
                    return np.column_stack([u ** a * v ** b for a, b in powers])

                def poly_fit(cuts):
                    """Coefficients of the polynomial fit to the cut-outs at (xs, ys)."""
                    A = basis(xs, ys)
                    Y = np.array([c[0] for c in cuts]).reshape(len(cuts), -1)
                    noise_s = np.array([c[1] for c in cuts])[:, None]
                    flux = np.array([c[2] for c in cuts])[:, None]
                    in_r = (rr_psf <= R).ravel()
                    coef = np.linalg.lstsq(A, Y, rcond=None)[0]   # unweighted start
                    # per-star inverse-variance weights from the start model, one refit: the mean of max(M, 0) over
                    # r <= R is about the model flux inside R / the pixel count for any model, so more rounds barely
                    # move them
                    M = A @ coef
                    var = noise_s ** 2 + (np.zeros_like(M) if gain is None else np.clip(M, 0, None) / (gain * flux))
                    sw = np.sqrt(1 / np.mean(var[:, in_r], axis=1))[:, None]
                    return np.linalg.lstsq(A * sw, Y * sw, rcond=None)[0]

                if gain is None:
                    print(f"[WARNING] {target_img.path.name}: no positive EGAIN; the polynomial PSF fit leaves the "
                          f"star Poisson noise out of its star weights")
                coef = poly_fit(cuts)
                tx, ty = np.meshgrid(xc, yc)   # (ny, nx) tile centres: row j, column i
                basis_tiles = basis(tx.ravel(), ty.ravel())
                p_tiles = (basis_tiles @ coef).reshape(gy, gx, n, n)
                for j in range(gy):
                    for i in range(gx):
                        psf[j, i] = finish(p_tiles[j, i])
                nstars[:] = len(cuts)
                self.helper.print(f"[PSFPhotometry] ePSF R={R} px, FWHM={fwhm:.2f} px, polynomial order {order} "
                                  f"from {len(cuts)} stars, evaluated on {gx} x {gy} (nx x ny) tiles", verbose)
            else:   # no usable order: model 'grid' with grid 1 from the same stars
                model, gx, gy = 'grid', 1, 1
                p, src = median_tile([c[0] for c in cuts], 'the image')
                psf = p[None, None]
                source = np.full((1, 1), src, dtype='<U6')
                nstars = np.full((1, 1), len(cuts), dtype=int)
                snr_used = snr_used[:1, :1]
                tile_cuts[0, 0] = cuts
        if model == 'grid':
            self.helper.print(f"[PSFPhotometry] ePSF R={R} px, FWHM={fwhm:.2f} px, stars per tile "
                              f"{nstars.ravel().tolist()}", verbose)

        # recenter_iter: star centres fitted with the current PSF (first moment at the array centre), PSF re-made
        recenter = []
        for it in range(recenter_iter):
            moves = []
            if model == 'poly':
                A = basis(xs, ys)
                for m, c in enumerate(cuts):
                    cuts[m], move = recentre(c, anchor(finish((A[m] @ coef).reshape(n, n))))
                    moves.append(move)
                coef = poly_fit(cuts)
                p_tiles = (basis_tiles @ coef).reshape(gy, gx, n, n)
                for j in range(gy):
                    for i in range(gx):
                        psf[j, i] = anchor(finish(p_tiles[j, i]))
            else:
                for (j, i), tcuts in tile_cuts.items():
                    if source[j, i] != 'epsf':   # Moffat tile: not re-made
                        continue
                    ref = anchor(psf[j, i])
                    for m, c in enumerate(tcuts):
                        tcuts[m], move = recentre(c, ref)
                        moves.append(move)
                    psf[j, i] = anchor(finish(np.median(np.array([c[0] for c in tcuts]), axis=0)))
            moves = np.array(moves, dtype=float)
            fitted = moves[np.isfinite(moves)]
            recenter.append(dict(n_fit=len(fitted), n_kept=len(moves) - len(fitted),
                                 median_shift=float(np.median(fitted)) if len(fitted) else np.nan,
                                 max_shift=float(fitted.max()) if len(fitted) else np.nan))
            self.helper.print(f"[PSFPhotometry] ePSF recentring round {it + 1}/{recenter_iter}: {len(fitted)} star "
                              f"centres fitted (shift median {recenter[-1]['median_shift']:.3f}, max "
                              f"{recenter[-1]['max_shift']:.3f} px), {len(moves) - len(fitted)} kept", verbose)
        if model == 'poly':
            psf_model = f'poly{order}'
        else:
            psf_model = 'moffat' if (source == 'moffat').all() else 'grid'
        info = dict(source=source, nstars=nstars, fwhm_pixel=fwhm, radius_pixel=R,
                    grid=grid if np.ndim(grid) == 0 and (gx, gy) == _grid_xy(grid) else (gx, gy), model=model,
                    poly_order=order if model == 'poly' else None, psf_model=psf_model,
                    recenter_iter=int(recenter_iter), recenter=recenter, snr_lower_used=snr_used)
        return (psf[0, 0] if (gx, gy) == (1, 1) else psf), info

    def psf_photometry(self,
                       target_img: Union[ScienceImage, ReferenceImage],
                       x_arr: Union[float, list, np.ndarray],
                       y_arr: Union[float, list, np.ndarray],
                       psf: np.ndarray = None,
                       psf_info: dict = None,
                       unit: str = 'coord',
                       target_bkg: Optional[Background] = None,
                       target_bkgrms: Optional[Errormap] = None,
                       target_mask: Optional[Mask] = None,
                       variance: np.ndarray = None,
                       noise_ratio: float = None,
                       host_img: Union[ScienceImage, ReferenceImage] = None,
                       background: str = 'annulus',
                       annulus_radii_arcsec: tuple = None,
                       fit_radius_fwhm: float = 2.5,
                       weighting: str = 'uniform',
                       zp_psf: float = None,
                       zperr_psf: float = None,
                       save: bool = True,
                       verbose: bool = True,
                       recenter: Union[bool, str] = False,
                       recenter_snr: float = 50.0,
                       recenter_max_shift: float = 0.5,
                       bkg_radius_fwhm: float = None,
                       neighbor_x: Union[list, np.ndarray] = None,
                       neighbor_y: Union[list, np.ndarray] = None,
                       neighbor_flux: Union[list, np.ndarray] = None,
                       neighbor_extended: Union[list, np.ndarray] = None,
                       max_neighbors: int = 6):
        """
        Forced PSF photometry: at each fixed position, a linear least-squares fit of the PSF amplitude, i.e. the total
        flux of the PSF array (normalised to sum 1 after the sub-pixel shift).

        Per position (x, y): the PSF at (x, y) (tiles blended with tent weights), shifted by the sub-pixel offset from
        the nearest pixel (scipy.ndimage.shift, order 3) and normalised to sum 1 = P. Fit pixels: within
        fit_radius_fwhm x FWHM (header seeing) of (x, y) and within the PSF array (with background 'fit' or 'plane'
        and bkg_radius_fwhm: within bkg_radius_fwhm x FWHM, also beyond the PSF array, where P = 0); pixels off the
        image, masked (target_mask), invalid (not finite or nonzero with |value| < 1e-20; an exact 0 is valid;
        ADU-scale values assumed) or without a positive finite weighting variance get weight 0.
        Model d = F P (+ C with background='fit', + C + Cx (x - x0) + Cy (y - y0) with 'plane'), solved by the
        weighted normal equations with w = 1 (weighting 'uniform': F = sum(P d) / sum(P^2), the PSF-weighted flux of a
        constant-variance fit; the variance only enters the errors and CHI2) or w = 1 / V_w ('inverse_variance').
        V_w leaves out the Poisson noise of the measured source itself (default variance): weights taken from the
        noisy data would down-weight pixels that fluctuate up and bias the flux low. Error from the per-pixel
        variance V:
        Var(F) = sum(w^2 P^2 V) / (sum(w P^2))^2 (the general weighted least-squares covariance with the background
        terms).
        Negative fluxes are kept.

        recenter='offset': the positions of the sources at or above recenter_snr are fitted as with 'each' (below,
        a first pass) and a smooth offset over the image is fitted to those offsets (per axis a + b u + c v, u, v = x, y
        scaled to [-1, 1] over the image, one 3-sigma (MAD) clipping and a refit; from >= 10 sources, the median from
        >= 3, else no offset with a warning); that offset is added to every position (all sources, whatever their
        S/N), and the fit above runs at the shifted positions (fit pixels, tile blend and annulus there).
        X_IMAGE/Y_IMAGE and X_WORLD/Y_WORLD stay the input positions; DX_PSF, DY_PSF hold the applied offset (px) of
        every source; FLAG_PSF bit 16 when it exceeds recenter_max_shift px. In this mode bit 16 is an image-level
        warning, not a bad fit: the offset is one model for the image, so the bit is set on many rows of that image
        at once, and keeping only FLAG_PSF == 0 can drop most of its sources. The meta RECENTER line has the
        coefficients, the sources used and the residual rms. No S/N step between sources: all positions move by the
        same smooth model. The offset is that of the PSF centres from the input positions of the measured sources:
        a smooth difference of centre definition or coordinate round trip; it includes a WCS error only when the
        input positions are sky-catalog coordinates and so are those of the measured sources (input positions
        measured on this image, e.g. its own source catalog, already hold the WCS error, which then stays).
        recenter='each' (or True): a source with F / FLUXERR_PSF >= recenter_snr from the fit above also gets its
        position fitted with the same PSF (Gauss-Newton on F, the sub-pixel offset (and the background terms of 'fit'
        or 'plane'), same pixels and weights, at most the fit radius away), and the fit above is redone at the fitted
        position (all columns from it; the fit pixels stay those around the input position, the PSF tile blend that
        of the input position).
        X_IMAGE/Y_IMAGE and X_WORLD/Y_WORLD stay the input positions; DX_PSF, DY_PSF hold the fitted offset (px). A
        move of more than recenter_max_shift px keeps the fitted-position values and sets FLAG_PSF bit 16; a failed
        position fit keeps the input-position values, DX_PSF/DY_PSF NaN and bit 16. Sources below recenter_snr (or
        without a fit): DX_PSF/DY_PSF NaN.

        Neighbours (neighbor_x, neighbor_y given; off by default): other sources (e.g. the image's own SExtractor
        catalog, 0-based X_IMAGE/Y_IMAGE) at fixed positions (recenter offsets are not added to them) are fitted
        together with a source. Per source (at its fit position): the given positions within the fit-pixel radius + 2
        FWHM of it except the nearest one within 1 FWHM (taken as the source's own detection, extended or not; if the
        source itself is not in the list, the nearest other source within 1 FWHM is dropped so) and any other closer
        than 0.5 FWHM (not separable: left out, neither fitted nor flagged; between 0.5 and 1 FWHM they are neighbours);
        extended ones (neighbor_extended) are not fitted and set FLAG_PSF bit 32; of the others the max_neighbors
        brightest (neighbor_flux; without it the nearest) get a PSF column (the PSF at their position, sub-pixel
        shifted, sum 1) in the same linear fit, after the background terms, with their own fitted amplitudes (not
        written out; NNEIGHBOR_PSF counts them). Fit pixels: the union of the fit circles (fit-pixel radius) of the
        source and its fitted neighbours (NPIX_PSF; CHI2_PSF counts the neighbour amplitudes as fitted parameters);
        NMASK_PSF and bits 2 and 4 stay those of the source's own circle; the default variance's Poisson term uses the
        source and neighbour models; FLUXERR_PSF is the source's marginal error (neighbour covariance included). With
        recenter 'each' the neighbour columns stay at their positions in the position fit; with 'offset' the first pass
        fits them too. A source missing from the list shows in CHI2_PSF. Without neighbours the results do not change.

        Background (the data minus target_bkg first, when given):
            'annulus' (default): minus the 5-sigma-clipped mean (5 iterations) of an annulus around the position
                (the estimator of AperturePhotometry.circular_photometry with annulus_width_arcsec; here masked and
                invalid pixels are left out of it), then F alone.
            'fit': F and a constant C together.
            'plane': F and a plane C + Cx (x - x0) + Cy (y - y0) together ((x0, y0) the source position, pixels;
                BACKGROUND = C, with recenter 'each' at the fitted position). Over fit pixels centred on the source
                the x, y terms (odd about the source) are nearly orthogonal to the constant and to a near
                point-symmetric P (even), so with no pixel left out 'plane' and 'fit' give almost the same flux and
                error, also under a pure gradient; the plane matters where the fit pixels are lopsided (masked or
                invalid pixels, the image edge) and for the position fits of recenter (a gradient pulls a fit with a
                constant only).
                On a galaxy ('fit' or 'plane'): the local level under the source is followed better than with the
                annulus, but galaxy structure the model does not follow within the fit pixels (curvature: a nucleus,
                a bulge or arm peaked at or near the source) is partly fitted as PSF flux and biases it: bright where
                the light curves down (on a peak, within about the inflection radius of a nucleus or bulge), faint
                where it curves up (outside it on the flank, between arms); the bias grows with the radius
                (bkg_radius_fwhm), the noise falls with it. Beyond the PSF
                array (bkg_radius_fwhm x FWHM > its radius) the source's own light, if any, goes into the background.
            'none': no local background. Without target_bkg the sky then stays in d and is not in the default
                variance's Poisson term (which uses the fitted PSF model), only in bkgrms^2.
        Variance (ADU^2 per pixel):
            variance given: V = V_w = variance (e.g. the variance plane of the image, or a difference-image noise map
                squared). With weighting 'inverse_variance' a variance that holds the measured source's own Poisson
                noise (a variance plane made from the image itself) weights by the data and biases the flux low.
            else: V_w = noise_ratio^2 x bkgrms^2 (+ max(host net, 0) / EGAIN with host_img) and
                V = V_w + max(F P, 0) / EGAIN: the Poisson noise of the source from the fitted PSF model, not from
                the data. (max(d - background, 0) / EGAIN, the calc_total_error form, adds sigma_pix / (sqrt(2 pi)
                EGAIN) per pixel from clipping pure noise at 0, which inflates the error of background-limited
                sources: +13 % variance at EGAIN 0.256, sigma_pix 12 ADU.) bkgrms = target_bkgrms, or computed
                from the image (ErrormapGenerator, not saved).

        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The image to measure (header SEEING; EGAIN for the default variance). Its files are only read; without
            target_bkgrms the in-memory header gets TGTPATH from ErrormapGenerator (pre-existing behaviour).
        x_arr, y_arr : float or array-like
            Positions: RA/Dec in deg (unit 'coord') or 0-based pixel (unit 'pixel').
        psf : np.ndarray, optional
            PSF of the image, (n, n) or (ny, nx, n, n) with n odd (see the class docstring). None: build_epsf
            with its defaults (and target_mask).
        psf_info : dict, optional
            The info of build_epsf for this PSF: FLAG_PSF bit 1 and the catalog meta use its 'source' (and 'model',
            'poly_order').
        unit : str, optional
            'coord' (default) or 'pixel'.
        target_bkg : Background, optional
            Subtracted from the data first (in memory; no file is written).
        target_bkgrms : Errormap, optional
            Background RMS for the default variance.
        target_mask : Mask, optional
            Nonzero pixels get weight 0 and stay out of the annulus (an exact 0 is valid data: pass 0-filled empty
            areas, e.g. the SWarp no-weight edges of DIA sci_ images written before 0.9.0, here; from 0.9.0 the
            reprojection sets them to NaN).
        variance : np.ndarray, optional
            Per-pixel variance (ADU^2) with the image's shape. None (default): the bkgrms and Poisson model above.
        noise_ratio : float, optional
            Default variance only: bkgrms^2 x noise_ratio^2 (e.g. a measured / predicted empty-aperture noise
            ratio). None: 1.
        host_img : ScienceImage or ReferenceImage, optional
            Default variance only: an image on the same grid and flux scale holding the light under the sources that
            target_img lacks (for a difference image the convolved reference); max(its pixel value minus its own
            annulus mean (background 'annulus', 'fit' and 'plane'), 0) / EGAIN is added to V and V_w.
        background : str, optional
            'annulus' (default), 'fit', 'plane' or 'none'.
        annulus_radii_arcsec : tuple, optional
            (inner, outer) annulus radius in arcsec. Defaults to the difference-image annulus of the forced_photometry
            recipe with its default apertures: inner = (max(15", 3.5 seeing) + 1.5 seeing) / 2, outer = inner + 30".
        fit_radius_fwhm : float, optional
            Fit radius in units of the FWHM (capped at the PSF array radius). Defaults to 2.5.
        weighting : str, optional
            'uniform' (default) or 'inverse_variance'.
        zp_psf, zperr_psf : float, optional
            PSF zero point to apply (e.g. calibrate_zp of the science image, for its difference image): adds
            MAGSKY_PSF, ZP_PSF, ZPERR_PSF (NaN without zperr_psf), UL3SKY_PSF, UL5SKY_PSF.
        save : bool, optional
            Write the catalog (default). False: nothing is written and no file is removed.
        verbose : bool, optional
            Whether to print.
        recenter : bool or str, optional
            False (default, fixed positions), 'offset' (one smooth offset of all positions from the position fits of
            the sources at or above recenter_snr) or 'each' (= True; the position of each source at or above
            recenter_snr fitted on its own) (see above).
        recenter_snr : float, optional
            S/N (F / FLUXERR_PSF of the fixed-position fit) from which a source's position is fitted. Defaults to 50.
        recenter_max_shift : float, optional
            Fitted move (px) beyond which FLAG_PSF bit 16 is set. Defaults to 0.5.
            With 'offset' the applied offset is compared instead (an image-level warning on many rows, see above).
        bkg_radius_fwhm : float, optional
            Background 'fit' or 'plane' only (else ValueError): radius of the fit pixels in units of the FWHM, in place
            of fit_radius_fwhm and not capped at the PSF array radius: the stamp is enlarged and the pixels beyond the
            array (P = 0) constrain the background only. NPIX_PSF, NMASK_PSF, FLAG_PSF bits 2 and 4 and the position
            fit of recenter use these pixels; the recenter move limit stays the fit radius. None (default):
            fit_radius_fwhm (the same results as without it).
        neighbor_x, neighbor_y : array-like, optional
            0-based pixel positions of other sources on target_img (whatever unit) to fit with each source (see
            Neighbours above). None (default): no neighbours, the results as without them. Non-finite positions are
            dropped.
        neighbor_flux : array-like, optional
            Brightness of those sources (e.g. FLUX_AUTO), for choosing the max_neighbors brightest; None: the nearest.
        neighbor_extended : array-like of bool, optional
            True for an extended source: not fitted, FLAG_PSF bit 32 on the sources near it. None: all point sources.
        max_neighbors : int, optional
            Neighbours fitted per source at most. Defaults to 6.

        Returns
        -------
        target_catalog : Catalog
            <img>.fits.psf.cat in the image's savedir (target_img.savepath.catalogpath with suffix .psf.cat), one row
            per position in the given order: X_IMAGE, Y_IMAGE (0-based pixel), X_WORLD, Y_WORLD (deg), FLUX_PSF,
            FLUXERR_PSF (ADU), MAG_PSF, MAGERR_PSF (instrumental; NaN unless FLUX_PSF > 0), UL3_PSF, UL5_PSF
            (instrumental 3 / 5 sigma of the V_w part of the error), BACKGROUND (local level per pixel subtracted or
            fitted, ADU), CHI2_PSF (sum((d - model)^2 / V) / (NPIX_PSF - fitted parameters); about 1 for a point
            source the PSF fits, values >~ 3 are worth a look (an empirical guide, not a flag): a blend, an extended
            source, a PSF mismatch or a position offset), NPIX_PSF (fit pixels used), NMASK_PSF (fit-radius pixels with
            weight 0), PSF_EE (sum of P over the fit pixels used), FLAG_PSF (bits: 1 Moffat fallback, 2 pixels with
            weight 0, 4 image edge, 8 no fit: the values NaN ('plane': also when the used pixels span fewer than 2
            columns or rows), 16 recenter: moved > recenter_max_shift or position fit
            failed; 'offset': applied offset > recenter_max_shift, an image-level warning on many rows at once, see
            above; with neighbours 32 an extended neighbour not fitted), with recenter also DX_PSF, DY_PSF (px), with
            neighbours NNEIGHBOR_PSF (neighbours fitted, after NMASK_PSF).
            The table meta 'comments' (written as '#' lines) records the settings.
        """
        if background not in ('annulus', 'fit', 'plane', 'none'):
            raise ValueError(f"background must be 'annulus', 'fit', 'plane' or 'none', got {background!r}")
        if bkg_radius_fwhm is not None and not (background in ('fit', 'plane') and np.isfinite(bkg_radius_fwhm)
                                                and bkg_radius_fwhm > 0):
            raise ValueError(f"bkg_radius_fwhm must be finite, > 0 and is for background 'fit' or 'plane', got "
                             f"{bkg_radius_fwhm!r} with {background!r}")
        if weighting not in ('uniform', 'inverse_variance'):
            raise ValueError(f"weighting must be 'uniform' or 'inverse_variance', got {weighting!r}")
        if unit not in ('pixel', 'coord'):
            raise ValueError("unit must be either 'pixel' or 'coord'")
        mode = ('each' if recenter else False) if isinstance(recenter, (bool, np.bool_)) else recenter
        if mode not in (False, 'offset', 'each'):
            raise ValueError(f"recenter must be False, 'offset', 'each' or True, got {recenter!r}")
        if psf is None:
            psf, psf_info = self.build_epsf(target_img, target_mask=target_mask, verbose=verbose)
        psf = np.asarray(psf, dtype=float)
        _check_psf(psf, 'psf')
        R = psf.shape[-1] // 2
        fwhm, scale = self._fwhm_pixel(target_img)
        fit_radius = min(fit_radius_fwhm * fwhm, R)
        # fit pixels within pix_radius; with bkg_radius_fwhm also beyond the PSF array (P = 0 there: background only),
        # in a stamp of half-width Rb >= R holding every pixel within pix_radius of a position up to 0.5 px off centre
        pix_radius = fit_radius if bkg_radius_fwhm is None else bkg_radius_fwhm * fwhm
        Rb = max(R, int(np.floor(pix_radius + 0.5)))
        pad = Rb - R

        # Neighbours (0-based pixel positions of other sources, fixed): fitted with a source within nb_radius of it
        nb_on = neighbor_x is not None
        nb_tree = None
        if nb_on:
            if neighbor_y is None:
                raise ValueError("neighbor_x needs neighbor_y")
            nb_x = np.atleast_1d(np.asarray(neighbor_x, dtype=float))
            nb_y = np.atleast_1d(np.asarray(neighbor_y, dtype=float))
            nb_ext = (np.zeros(nb_x.shape, dtype=bool) if neighbor_extended is None
                      else np.atleast_1d(np.asarray(neighbor_extended, dtype=bool)))
            nb_flux = None if neighbor_flux is None else np.atleast_1d(np.asarray(neighbor_flux, dtype=float))
            if (nb_x.ndim != 1 or nb_y.shape != nb_x.shape or nb_ext.shape != nb_x.shape
                    or (nb_flux is not None and nb_flux.shape != nb_x.shape)):
                raise ValueError("neighbor_x, neighbor_y (and neighbor_flux, neighbor_extended) need the same length")
            if int(max_neighbors) != max_neighbors or max_neighbors < 0:
                raise ValueError(f"max_neighbors must be an int >= 0, got {max_neighbors!r}")
            ok = np.isfinite(nb_x) & np.isfinite(nb_y)
            nb_x, nb_y, nb_ext = nb_x[ok], nb_y[ok], nb_ext[ok]
            nb_flux = None if nb_flux is None else nb_flux[ok]
            nb_radius, nb_self = pix_radius + _NEIGHBOR_RADIUS_FWHM * fwhm, _NEIGHBOR_SELF_FWHM * fwhm
            nb_min = _NEIGHBOR_MIN_FWHM * fwhm
            if len(nb_x):
                nb_tree = cKDTree(np.column_stack([nb_x, nb_y]))
        nb_kwargs = dict(neighbor_x=neighbor_x, neighbor_y=neighbor_y, neighbor_flux=neighbor_flux,
                         neighbor_extended=neighbor_extended, max_neighbors=max_neighbors)

        # Data, invalid pixels (judged before any background subtraction) and mask
        raw = self.helper.to_native(target_img.data)
        bad = _bad_pixels(raw)
        if target_mask is not None:
            bad |= np.asarray(self.helper.to_native(target_mask.data)).astype(bool)
        data = raw if target_bkg is None else raw - self.helper.to_native(target_bkg.data)
        ny, nx = raw.shape

        # Variance inputs
        egain = target_img.egain
        bkgrms = host = None
        if variance is not None:
            variance = np.asarray(variance)
            if variance.shape != raw.shape:
                raise ValueError(f"variance has shape {variance.shape}, the image {raw.shape}")
        else:
            if egain is None or not egain > 0:
                raise ValueError(f"{target_img.path.name}: no positive EGAIN for the Poisson term; give variance=")
            if target_bkgrms is None:
                target_bkgrms, _, _ = self.errormap.calculate_errormap_from_image(
                    target_img = target_img,
                    errormap_type = 'bkgrms',
                    save = False,
                    verbose = verbose,
                    visualize = False,
                    save_fig = False
                )
            bkgrms = self.helper.to_native(target_bkgrms.data)
            if host_img is not None:
                host = self.helper.to_native(host_img.data)
        ratio2 = 1.0 if noise_ratio is None else float(noise_ratio) ** 2

        # Positions
        x_arr = np.atleast_1d(np.asarray(x_arr, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y_arr, dtype=float))
        skycoord = None
        if unit == 'pixel':
            x_pix, y_pix = x_arr, y_arr
            if target_img.wcs:
                skycoord = pixel_to_skycoord(x_pix, y_pix, target_img.wcs, origin=0)
        else:
            skycoord = SkyCoord(ra=x_arr, dec=y_arr, unit='deg')
            x_pix, y_pix = skycoord_to_pixel(skycoord, target_img.wcs, origin=0)
            x_pix, y_pix = np.atleast_1d(np.asarray(x_pix, dtype=float)), np.atleast_1d(np.asarray(y_pix, dtype=float))
        nsrc = len(x_pix)
        x_in, y_in = x_pix, y_pix   # X_IMAGE/Y_IMAGE of the catalog: the input positions
        if mode == 'offset':
            # first pass: position fits of the sources at or above recenter_snr ('each'), then one smooth offset of all
            measured = self.psf_photometry(
                target_img, x_pix, y_pix, psf=psf, psf_info=psf_info, unit='pixel', target_bkg=target_bkg,
                target_bkgrms=target_bkgrms, target_mask=target_mask, variance=variance, noise_ratio=noise_ratio,
                host_img=host_img, background=background, annulus_radii_arcsec=annulus_radii_arcsec,
                fit_radius_fwhm=fit_radius_fwhm, weighting=weighting, save=False, verbose=False, recenter='each',
                recenter_snr=recenter_snr, bkg_radius_fwhm=bkg_radius_fwhm, **nb_kwargs).data
            off_x, off_y, n_off, n_meas, off_rms = _offset_model(x_pix, y_pix, np.asarray(measured['DX_PSF']),
                                                                 np.asarray(measured['DY_PSF']), raw.shape)
            if n_off == 0:
                print(f"[WARNING] {target_img.path.name}: {n_meas} sources with a position fit (< "
                      f"{_OFFSET_MIN_CONST}); no recenter offset applied")
            u, v = 2 * x_pix / (nx - 1) - 1, 2 * y_pix / (ny - 1) - 1
            offset = np.array([off_x[0] + off_x[1] * u + off_x[2] * v, off_y[0] + off_y[1] * u + off_y[2] * v])
            x_pix, y_pix = x_pix + offset[0], y_pix + offset[1]
        # The PSF array overlaps the image
        on_image = np.isfinite(x_pix) & np.isfinite(y_pix)
        on_image &= (x_pix > -R - 0.5) & (x_pix < nx - 0.5 + R) & (y_pix > -R - 0.5) & (y_pix < ny - 0.5 + R)

        # Local background (and the host image's own level)
        local = np.zeros(nsrc)
        host_local = np.zeros(nsrc)
        r_in = r_out = None
        if background == 'annulus' or (host is not None and background in ('fit', 'plane')):
            seeing = fwhm * scale
            if annulus_radii_arcsec is None:
                r_in = (max(15.0, 3.5 * seeing) + 1.5 * seeing) / 2
                r_out = r_in + 30.0
            else:
                r_in, r_out = (float(r) for r in annulus_radii_arcsec)
            idx = np.where(on_image)[0]
            if len(idx):
                annulus = CircularAnnulus(np.column_stack([x_pix[idx], y_pix[idx]]),
                                          r_in=r_in / scale, r_out=r_out / scale)
                sigma_clip = SigmaClip(sigma=5.0, maxiters=5)
                if background == 'annulus':
                    local[idx] = np.atleast_1d(ApertureStats(data, annulus, sigma_clip=sigma_clip, mask=bad).mean)
                if host is not None:
                    host_local[idx] = np.atleast_1d(ApertureStats(host, annulus, sigma_clip=sigma_clip, mask=bad).mean)

        # Fit
        cols = {key: np.full(nsrc, np.nan) for key in
                ('FLUX_PSF', 'FLUXERR_PSF', 'FLUXERR_BKG', 'BACKGROUND', 'CHI2_PSF', 'PSF_EE')}
        npix_used = np.zeros(nsrc, dtype=int)
        nmask = np.zeros(nsrc, dtype=int)
        flags = np.zeros(nsrc, dtype=int)
        n_par = {'fit': 2, 'plane': 4}.get(background, 1)
        n_box = 2 * Rb + 1
        yy, xx = np.indices((n_box, n_box))
        fallback = None
        if psf_info is not None and 'source' in psf_info:
            fallback = np.atleast_2d(np.asarray(psf_info['source'])) == 'moffat'
        if psf.ndim == 4:
            txc, tyc, _, _ = _tiles(raw.shape, (psf.shape[1], psf.shape[0]))
        dxy = np.full((2, nsrc), np.nan)   # recenter: fitted offset from the input position (px)
        nneigh = np.zeros(nsrc, dtype=int)   # neighbours fitted with each source

        def psf_at(x, y):
            """The (unshifted) PSF at (x, y): tiles blended with tent weights."""
            if psf.ndim == 2:
                return psf
            return np.einsum('j,i,jimn->mn', _tent_weights(y, tyc), _tent_weights(x, txc), psf)

        def flux_fit(P, D, W, Vw, B, N=None):
            """Weighted linear fit of D = F P (+ B @ c, background columns B) (+ N @ f, neighbour PSF columns N, last):
            theta, V, cov, cov_bkg, resid (LinAlgError when singular)."""
            X = np.column_stack([P, B]) if B is not None else P[:, None]
            if N is not None:
                X = np.column_stack([X, N])
            Minv = np.linalg.inv(X.T @ (W[:, None] * X))
            theta = Minv @ (X.T @ (W * D))
            # Source Poisson from the fitted model: clipping the noisy data at 0 would bias V up for faint sources
            if N is None:
                V = Vw if variance is not None else Vw + np.clip(theta[0] * P, 0, None) / egain
            else:   # the neighbours' fitted models too
                V = Vw if variance is not None else Vw + np.clip(theta[0] * P + N @ theta[-N.shape[1]:], 0,
                                                                 None) / egain
            cov = Minv @ (X.T @ ((W ** 2 * V)[:, None] * X)) @ Minv
            cov_bkg = Minv @ (X.T @ ((W ** 2 * Vw)[:, None] * X)) @ Minv
            return theta, V, cov, cov_bkg, D - X @ theta

        for k in range(nsrc):
            if not on_image[k]:
                flags[k] |= FLAG_FAILED
                continue
            # PSF at the position, shifted by the sub-pixel offset from the nearest pixel
            if psf.ndim == 4:
                wy, wx = _tent_weights(y_pix[k], tyc), _tent_weights(x_pix[k], txc)
                p = np.einsum('j,i,jimn->mn', wy, wx, psf)
                if (fallback is not None and fallback.shape == (len(wy), len(wx))
                        and (fallback & (np.outer(wy, wx) > 0)).any()):
                    flags[k] |= FLAG_FALLBACK
            else:
                p = psf
                if fallback is not None and fallback.any():
                    flags[k] |= FLAG_FALLBACK
            xi, yi = int(np.round(x_pix[k])), int(np.round(y_pix[k]))
            dx, dy = x_pix[k] - xi, y_pix[k] - yi
            p0 = p   # unshifted, for recenter
            p = ndimage.shift(p, (dy, dx), order=3, mode='constant')
            p = p / p.sum()
            # Neighbours of this source (within nb_radius but the nearest within nb_self, its own detection, and those
            # closer than nb_min; extended ones only flagged), the brightest (or nearest) max_neighbors; the stamp
            # (half-width Rk) grows to hold their PSF arrays
            sel = []
            if nb_tree is not None:
                near = np.asarray(nb_tree.query_ball_point([x_pix[k], y_pix[k]], nb_radius), dtype=int)
                dist = np.hypot(nb_x[near] - x_pix[k], nb_y[near] - y_pix[k])
                if len(near) and dist.min() <= nb_self:   # extended or not: that one is the source itself
                    near = np.delete(near, np.argmin(dist))
                near = near[np.hypot(nb_x[near] - x_pix[k], nb_y[near] - y_pix[k]) >= nb_min]   # closer: not separable
                if nb_ext[near].any():
                    flags[k] |= FLAG_NEIGHBOR_EXT
                near = near[~nb_ext[near]]
                order = (np.argsort(-nb_flux[near], kind='stable') if nb_flux is not None else
                         np.argsort(np.hypot(nb_x[near] - x_pix[k], nb_y[near] - y_pix[k]), kind='stable'))
                sel = near[order][:int(max_neighbors)]
                nneigh[k] = len(sel)
            Rk, xk, yk = Rb, xx, yy
            if len(sel):
                Rk = Rb + int(np.ceil(np.max(np.abs(np.concatenate([nb_x[sel] - x_pix[k], nb_y[sel] - y_pix[k]]))))) + 1
                yk, xk = np.indices((2 * Rk + 1, 2 * Rk + 1))
            padk, nbox = Rk - R, 2 * Rk + 1
            if padk:   # pixels beyond the PSF array: P = 0
                p = np.pad(p, padk)

            # Stamp around (xi, yi); pixels off the image are invalid
            x0, y0 = xi - Rk, yi - Rk
            sx = slice(max(x0, 0), min(x0 + nbox, nx))
            sy = slice(max(y0, 0), min(y0 + nbox, ny))
            ix = slice(sx.start - x0, sx.stop - x0)
            iy = slice(sy.start - y0, sy.stop - y0)
            inside = np.zeros((nbox, nbox), dtype=bool)
            inside[iy, ix] = True
            d = np.zeros((nbox, nbox))
            d[iy, ix] = data[sy, sx]
            b = np.ones((nbox, nbox), dtype=bool)
            b[iy, ix] = bad[sy, sx]
            vw = np.full((nbox, nbox), np.nan)
            if variance is not None:
                vw[iy, ix] = variance[sy, sx]
            else:
                vw[iy, ix] = ratio2 * np.asarray(bkgrms[sy, sx], dtype=float) ** 2
                if host is not None:
                    hv = np.zeros((nbox, nbox))
                    hv[iy, ix] = host[sy, sx]
                    vw = vw + np.clip(hv - host_local[k], 0, None) / egain
            footprint = np.hypot(xk - Rk - dx, yk - Rk - dy) <= pix_radius
            with np.errstate(invalid='ignore'):
                good = footprint & ~b & np.isfinite(vw) & (vw > 0)
            nmask[k] = int((footprint & ~good).sum())
            if nmask[k]:
                flags[k] |= FLAG_MASKED
            if (footprint & ~inside).any():
                flags[k] |= FLAG_EDGE
            n_own, N = int(good.sum()), None   # NMASK_PSF, bits 2 and 4: the source's own fit circle
            if len(sel):   # neighbour PSF stamps; fit pixels = the union of the fit circles of the source and them
                N = np.zeros((len(sel), nbox, nbox))
                union = footprint.copy()
                for j, n in enumerate(sel):
                    xn, yn = int(np.round(nb_x[n])), int(np.round(nb_y[n]))
                    pn = ndimage.shift(psf_at(nb_x[n], nb_y[n]), (nb_y[n] - yn, nb_x[n] - xn), order=3,
                                       mode='constant')
                    cy, cx = Rk + yn - yi, Rk + xn - xi
                    N[j, cy - R:cy + R + 1, cx - R:cx + R + 1] = pn / pn.sum()
                    union |= np.hypot(xk - Rk - (nb_x[n] - xi), yk - Rk - (nb_y[n] - yi)) <= pix_radius
                with np.errstate(invalid='ignore'):
                    good = union & ~b & np.isfinite(vw) & (vw > 0)
            npix_used[k] = int(good.sum())
            if (npix_used[k] <= n_par + len(sel) or n_own <= n_par or not np.isfinite(local[k])
                    or not np.isfinite(host_local[k])
                    or (background == 'plane' and (np.ptp(xk[good]) == 0 or np.ptp(yk[good]) == 0))):
                # ('plane': pixels in one column or row make the x or y term a multiple of the constant)
                flags[k] |= FLAG_FAILED
                continue

            P = p[good]
            D = d[good] - local[k]
            Vw = vw[good]
            W = np.ones_like(P) if weighting == 'uniform' else 1.0 / Vw
            B = None   # background columns: 1 ('fit'); 1, x - x0, y - y0 ('plane', pixels from the source position)
            if background == 'fit':
                B = np.ones_like(P)[:, None]
            elif background == 'plane':
                B = np.column_stack([np.ones_like(P), (xk - Rk - dx)[good], (yk - Rk - dy)[good]])
            Ng = None if N is None else N[:, good].T   # neighbour columns, after the background ones
            try:
                theta, V, cov, cov_bkg, resid = flux_fit(P, D, W, Vw, B, Ng)
            except np.linalg.LinAlgError:
                flags[k] |= FLAG_FAILED
                continue
            dof = npix_used[k] - n_par - len(sel)
            if (mode == 'each' and npix_used[k] > n_par + len(sel) + 2 and cov[0, 0] > 0
                    and theta[0] / np.sqrt(cov[0, 0]) >= recenter_snr):
                # position fit with the same PSF, pixels and weights (neighbours as fixed-position linear columns);
                # then the linear fit above at that position
                w_fit = np.zeros((nbox, nbox))
                w_fit[good] = W
                lin = B if Ng is None else (Ng if B is None else np.column_stack([B, Ng]))
                fit = _fit_shift(d - local[k], p0, w_fit, dx, dy, bkg=lin, limit=fit_radius)
                if fit is not None:
                    p1 = ndimage.shift(p0, (fit[1], fit[0]), order=3, mode='constant')
                    p1 = p1 / p1.sum()
                    P1 = (np.pad(p1, padk) if padk else p1)[good]
                    if background == 'plane':   # C at the fitted position
                        B = B - np.array([0.0, fit[0] - dx, fit[1] - dy])
                    try:
                        theta, V, cov, cov_bkg, resid = flux_fit(P1, D, W, Vw, B, Ng)
                        P, dof = P1, dof - 2
                        dxy[:, k] = fit[0] - dx, fit[1] - dy
                    except np.linalg.LinAlgError:   # keep the input-position fit
                        fit = None
                if fit is None or np.hypot(dxy[0, k], dxy[1, k]) > recenter_max_shift:
                    flags[k] |= FLAG_RECENTER
            level = theta[1] if n_par > 1 else 0.0
            cols['FLUX_PSF'][k] = theta[0]
            cols['FLUXERR_PSF'][k] = np.sqrt(cov[0, 0])
            cols['FLUXERR_BKG'][k] = np.sqrt(cov_bkg[0, 0])
            cols['BACKGROUND'][k] = local[k] + level
            cols['CHI2_PSF'][k] = float(np.sum(resid ** 2 / V)) / dof
            cols['PSF_EE'][k] = float(P.sum())

        # Catalog
        flux, flux_err, flux_err_bkg = cols['FLUX_PSF'], cols['FLUXERR_PSF'], cols['FLUXERR_BKG']
        results = Table()
        results['X_IMAGE'] = x_in
        results['Y_IMAGE'] = y_in
        if skycoord is not None:
            results['X_WORLD'] = np.atleast_1d(skycoord.ra.value)
            results['Y_WORLD'] = np.atleast_1d(skycoord.dec.value)
        results['FLUX_PSF'] = flux
        results['FLUXERR_PSF'] = flux_err
        mag = np.full(nsrc, np.nan)
        magerr = np.full(nsrc, np.nan)
        valid = flux > 0
        mag[valid] = -2.5 * np.log10(flux[valid])
        valid &= flux_err > 0
        magerr[valid] = 2.5 / np.log(10) * flux_err[valid] / flux[valid]
        results['MAG_PSF'] = mag
        results['MAGERR_PSF'] = magerr
        for n_sigma in (3, 5):
            ul = np.full(nsrc, np.nan)
            valid = flux_err_bkg > 0
            ul[valid] = -2.5 * np.log10(n_sigma * flux_err_bkg[valid])
            results[f'UL{n_sigma}_PSF'] = ul
        results['BACKGROUND'] = cols['BACKGROUND']
        results['CHI2_PSF'] = cols['CHI2_PSF']
        results['NPIX_PSF'] = npix_used
        results['NMASK_PSF'] = nmask
        if nb_on:
            results['NNEIGHBOR_PSF'] = nneigh
        results['PSF_EE'] = cols['PSF_EE']
        if mode == 'offset':   # the applied offset, for every source
            dxy = offset
            flags[np.hypot(offset[0], offset[1]) > recenter_max_shift] |= FLAG_RECENTER
        results['FLAG_PSF'] = flags
        if mode:
            results['DX_PSF'] = dxy[0]
            results['DY_PSF'] = dxy[1]
        psf_source = 'given'
        if psf_info is not None and 'source' in psf_info:
            src = np.asarray(psf_info['source'])
            model = psf_info.get('model', 'grid')
            if model == 'grid' and (src.ndim < 2 or src.shape[0] == src.shape[1]):
                psf_source = ' '.join(src.ravel().tolist())   # N x N tiles: one word per tile
            else:   # tiles per source, e.g. 'epsf x 228 (12 x 19 tiles, ny x nx; model poly order 3)'
                names, counts = np.unique(src, return_counts=True)
                if model == 'poly':
                    model = f"poly order {psf_info.get('poly_order')}"
                psf_source = (', '.join(f'{w} x {c}' for w, c in zip(names.tolist(), counts.tolist()))
                              + f' ({src.shape[0]} x {src.shape[1]} tiles, ny x nx; model {model})')
        results.meta['comments'] = [
            'ezphot PSFPhotometry.psf_photometry: X_IMAGE/Y_IMAGE are 0-based pixel coordinates',
            f'PSF_SHAPE = {psf.shape}',
            f'PSF_SOURCE = {psf_source}',
            f'FIT_RADIUS_PIX = {pix_radius:.3f}',
            f'BACKGROUND_MODE = {background}' + (f' {r_in:.2f}-{r_out:.2f} arcsec' if background == 'annulus' else '')
            + (f' within {pix_radius:.3f} px' if background == 'plane' or bkg_radius_fwhm is not None else ''),
            f'WEIGHTING = {weighting}',
            'VARIANCE = ' + ('given' if variance is not None else
                             f'bkgrms^2 x {ratio2:.4f} + Poisson' + (' + host' if host is not None else '')),
        ]
        if nb_on:
            results.meta['comments'].append(
                f'NEIGHBORS = {len(nb_x)} given positions (0-based px, fixed, not moved by recenter): those within '
                f'{nb_radius:.2f} px of a source but the nearest within {nb_self:.2f} px (its own detection) and '
                f'those within {nb_min:.2f} px (not separable) fitted with it, up to {int(max_neighbors)} '
                f'({"brightest neighbor_flux" if nb_flux is not None else "nearest"}; NNEIGHBOR_PSF), extended ones '
                f'not fitted (FLAG_PSF {FLAG_NEIGHBOR_EXT}); fit pixels the union of their fit circles (NPIX_PSF; '
                f'NMASK_PSF, FLAG_PSF {FLAG_MASKED} and {FLAG_EDGE} the source\'s own)')
        if mode == 'offset':
            results.meta['comments'].append(
                f'RECENTER = offset: DX_PSF = {off_x[0]:+.4f} {off_x[1]:+.4f} u {off_x[2]:+.4f} v, DY_PSF = '
                f'{off_y[0]:+.4f} {off_y[1]:+.4f} u {off_y[2]:+.4f} v px (u, v = X/Y_IMAGE scaled to [-1, 1] over the '
                f'image) from {n_off} of {n_meas} position fits at S/N >= {recenter_snr:g} (residual rms '
                f'{off_rms[0]:.3f} {off_rms[1]:.3f} px), added to all positions (X/Y_IMAGE, X/Y_WORLD are the input '
                f'positions), FLAG_PSF {FLAG_RECENTER} beyond {recenter_max_shift:g} px')
        elif mode == 'each':
            results.meta['comments'].append(
                f'RECENTER = S/N >= {recenter_snr:g}: position fitted for {int(np.isfinite(dxy[0]).sum())} of {nsrc} '
                f'(DX_PSF/DY_PSF px; X/Y_IMAGE, X/Y_WORLD are the input positions), FLAG_PSF {FLAG_RECENTER} beyond '
                f'{recenter_max_shift:g} px or position fit failed')
        if zp_psf is not None:
            self._add_zp_columns(results, zp_psf, zperr_psf)

        cat_path = target_img.savepath.catalogpath.with_suffix('.psf.cat')
        target_catalog = Catalog(path = cat_path, catalog_type = 'all', load = False, target_img = target_img)
        target_catalog.data = results
        if save:
            target_catalog.write(verbose = verbose)
        return target_catalog

    @staticmethod
    def _add_zp_columns(results: Table, zp: float, zperr: float):
        """MAGSKY_PSF, ZP_PSF, ZPERR_PSF, UL3SKY_PSF, UL5SKY_PSF columns for a given zero point and a meta line."""
        zp = float(zp)
        zperr = np.nan if zperr is None else float(zperr)
        results['MAGSKY_PSF'] = results['MAG_PSF'] + zp
        results['ZP_PSF'] = np.full(len(results), zp)
        results['ZPERR_PSF'] = np.full(len(results), zperr)
        for n_sigma in (3, 5):
            results[f'UL{n_sigma}SKY_PSF'] = results[f'UL{n_sigma}_PSF'] + zp
        results.meta.setdefault('comments', []).append(f'ZP_PSF = {zp:.5f} ZPERR_PSF = {zperr:.5f} (given)')

    def calibrate_zp(self,
                     target_img: Union[ScienceImage, ReferenceImage],
                     target_catalog: Catalog,
                     save: bool = True,
                     verbose: bool = True,
                     visualize: bool = False,
                     **kwargs):
        """
        PSF zero point from the reference stars measured in a psf_photometry catalog of the same image.

        target_catalog must hold the reference stars (e.g. psf_photometry at the targets and at the X_WORLD/Y_WORLD
        of <img>.fits.refcat in one call). PhotometricCalibration.photometric_calibration runs on it with
        magnitude_key='MAG_PSF', magnitudeerr_key='MAGERR_PSF': the same reference catalog matching, star
        selection (cuts on columns the catalog lacks are skipped, as for a forced aperture catalog) and ZP rule
        (3-sigma clipping, 5 iterations; ZP_PSF = median, ZPERR_PSF = std of the kept stars). It adds MAGSKY_PSF,
        ZP_PSF, ZPERR_PSF and, from UL3_PSF/UL5_PSF, UL3SKY_PSF/UL5SKY_PSF to every row. The image header and FITS
        are not changed (update_header=False, save_image=False) and no reference catalog is written
        (save_refcat=False).

        Parameters
        ----------
        target_img : ScienceImage or ReferenceImage
            The image of the catalog.
        target_catalog : Catalog
            psf_photometry catalog of target_img.
        save : bool, optional
            Write the catalog afterwards (default).
        verbose : bool, optional
            Whether to print.
        visualize : bool, optional
            Passed to photometric_calibration (its own default is True: figures shown). Defaults to False.
        **kwargs : dict
            Further photometric_calibration arguments (catalog_type, catalog_version, mag_lower, mag_upper,
            isolation_radius_arcsec, inner_fraction, ...). magnitude_key, magnitudeerr_key, update_header,
            save_image, save_refcat and save are fixed here (passing them raises ValueError).

        Returns
        -------
        target_catalog : Catalog
            The catalog with the ZP columns.
        zp : float
            ZP_PSF (NaN when none could be computed).
        zperr : float
            ZPERR_PSF.
        n_ref : int
            Reference stars that passed the selection.
        """
        fixed = ('magnitude_key', 'magnitudeerr_key', 'update_header', 'save_image', 'save_refcat', 'save')
        given = [k for k in fixed if k in kwargs]
        if given:
            raise ValueError(f"calibrate_zp fixes {given}")
        _, target_catalog, used, _ = PhotometricCalibration().photometric_calibration(
            target_img = target_img,
            target_catalog = target_catalog,
            magnitude_key = 'MAG_PSF',
            magnitudeerr_key = 'MAGERR_PSF',
            update_header = False,
            save_image = False,
            save_refcat = False,
            save = False,
            verbose = verbose,
            visualize = visualize,
            **kwargs)
        data = target_catalog.data
        zp = float(data['ZP_PSF'][0]) if 'ZP_PSF' in data.colnames and len(data) else np.nan
        zperr = float(data['ZPERR_PSF'][0]) if 'ZPERR_PSF' in data.colnames and len(data) else np.nan
        n_ref = int(used.nsources or 0)
        data.meta.setdefault('comments', []).append(
            f'ZP_PSF = {zp:.5f} ZPERR_PSF = {zperr:.5f} (reference stars: photometric_calibration on MAG_PSF, '
            f'NREF_PSF = {n_ref})')
        if save:
            target_catalog.write(verbose = verbose)
        return target_catalog, zp, zperr, n_ref
