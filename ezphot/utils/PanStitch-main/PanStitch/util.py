import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

import matplotlib.pyplot as plt
plt.ioff()
import matplotlib as mpl
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# When downloading images, coordinates in degrees is needed, and when using Swarp, coordinates in HMS DMS is needed
# If both format is prepared, no need of this code
# Just converting coordinate format
def degrees_to_hms_dms(ra_in_degrees, dec_in_degrees):
    """
    Convert RA in degrees to HMS (hours, minutes, seconds) and 
    Dec in degrees to DMS (degrees, minutes, seconds).
    
    Parameters:
    ra_in_degrees (float): RA in degrees.
    dec_in_degrees (float): Dec in degrees.

    Returns:
    tuple: The RA in HMS format (HH:MM:SS) and Dec in DMS format (±DD:MM:SS.ssss).
    """
    coord = SkyCoord(ra=ra_in_degrees * u.deg, dec=dec_in_degrees * u.deg, frame='icrs')
    
    ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':', pad=True, precision=0)  # HH:MM:SS
    dec_dms = coord.dec.to_string(unit=u.degree, sep=':', alwayssign=True, precision=4)  # ±DD:MM:SS.ssss
    
    return ra_hms, dec_dms

def generate_pointings(center_ra, center_dec, xsize, ysize, pixscale, n=8, m=8, margin_frac=0.1):
    """
    Generate RA, Dec pointings based on input center coordinates and image size.
    
    Parameters:
    - center_ra (float): Central RA coordinate in degrees.
    - center_dec (float): Central Dec coordinate in degrees.
    - xsize (int): X size of the image in pixels.
    - ysize (int): Y size of the image in pixels.
    - pixscale (float): Pixel scale in arcseconds/pixel.
    - n (int): Number of pointings along the X-axis (default is 8).
    - m (int): Number of pointings along the Y-axis (default is 8).
    - margin_frac (float): Fractional margin to add around the image (default is 10%).
    
    Returns:
    - List of tuples with (RA, Dec) for each pointing.
    """
    
    # Calculate total margin to cover the entire field of view
    xmargin = xsize * (1 + margin_frac)
    ymargin = ysize * (1 + margin_frac)

    # Convert pixel margins to degrees
    ra_range = xmargin * pixscale / 3600.0 / 2.0  # Convert arcsec to degrees
    dec_range = ymargin * pixscale / 3600.0 / 2.0

    # Generate evenly spaced RA and Dec points
    ra_points = np.linspace(center_ra - ra_range, center_ra + ra_range, n)
    dec_points = np.linspace(center_dec - dec_range, center_dec + dec_range, m)

    # Create all combinations of RA and Dec pointings
    pointings = [(ra, dec) for ra in ra_points for dec in dec_points]

    return pointings

def plot_rectangle_on_sky(center_ra, center_dec, x_size, y_size, pixel_scale, color='red'):
    """
    Plot a rectangle on the sky with a given center and size in arcseconds.
    
    Parameters:
    - center_ra (float): Central RA in degrees.
    - center_dec (float): Central Dec in degrees.
    - x_size (int): X size of the image in pixels.
    - y_size (int): Y size of the image in pixels.
    - pixel_scale (float): Pixel scale in arcseconds per pixel.
    - color (str): Color of the rectangle.
    """
    # Convert size from pixels to degrees
    x_deg = x_size * pixel_scale / 3600.0
    y_deg = y_size * pixel_scale / 3600.0
    
    # Plot the rectangle with four corners
    ra_min = center_ra - x_deg / 2
    ra_max = center_ra + x_deg / 2
    dec_min = center_dec - y_deg / 2
    dec_max = center_dec + y_deg / 2
    
    # Plot the rectangle (as a series of lines between the corners)
    plt.plot([ra_min, ra_max], [dec_min, dec_min], color=color)  # Bottom edge
    plt.plot([ra_min, ra_max], [dec_max, dec_max], color=color)  # Top edge
    plt.plot([ra_min, ra_min], [dec_min, dec_max], color=color)  # Left edge
    plt.plot([ra_max, ra_max], [dec_min, dec_max], color=color)  # Right edge

def plot_pointings_on_sky(pointings, center_ra, center_dec, x_size, y_size, pixel_scale, target_outpath, title=None, save=True):
    """
    Plot the pointings and the central rectangle on the sky.
    
    Parameters:
    - pointings (list of tuples): List of (RA, Dec) for each pointing.
    - center_ra (float): Central RA in degrees.
    - center_dec (float): Central Dec in degrees.
    - x_size (int): X size of the image in pixels.
    - y_size (int): Y size of the image in pixels.
    - pixel_scale (float): Pixel scale in arcseconds per pixel.
    - target_outpath (str): Path to save the plot image.
    - title (str, optional): Title of the plot.
    - save (bool, optional): Whether to save the plot (default is True).
    """
    
    # Calculate the aspect ratio from x_size and y_size
    aspect_ratio = x_size / y_size
    
    # Multiply by a scaling factor (4 in this case) to make the figure large enough
    scaling_factor = 4
    figsize_x = aspect_ratio * scaling_factor
    figsize_y = scaling_factor
    
    # Set the figure size based on the aspect ratio
    plt.figure(figsize=(figsize_x, figsize_y))

    # Plot all pointings
    ra_vals, dec_vals = zip(*pointings)
    plt.plot(ra_vals, dec_vals, marker='.', ls='none', color='grey', label='Pointings')
    
    # Plot the center and the rectangle
    plt.plot(center_ra, center_dec, 'rx', ms=10, label='Center')
    plot_rectangle_on_sky(center_ra, center_dec, x_size, y_size, pixel_scale, color='red')
    
    # Labels and title
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    if title:
        plt.title(title)

    # Legend
    plt.legend(loc='upper center')
    plt.tight_layout()
    # Save or show the plot
    if save:
        plt.savefig(target_outpath)
        print(f"Plot saved to {target_outpath}")
    # else:
        # plt.show()
    plt.show()
    plt.close('all')