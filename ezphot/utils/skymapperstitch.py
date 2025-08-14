#%%
import numpy as np
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

import matplotlib.pyplot as plt
plt.ioff()
import matplotlib as mpl
import requests
from astropy.io.votable import parse
import io
import pandas as pd
import os
import time
#%%

class SkyMapperStitch:
    """
    Class to stitch SkyMapper images together.
    Author: Martin Moonkuk Hur
    Modified by: Hyeonho Choi
  
    """

    def __init__(self):
        """
        Initialize the SkyMapperStitch class with a list of images.
        
        :param images: List of SkyMapper image objects.
        """
        
    def write_images_to_swarp(list_file_path, image_files):
        """
        Create the file list for SWarp input.
        
        Parameters:
        - list_file_path (str): Path to save the list file.
        - image_files (list): List of masked image file paths.
        """
        with open(list_file_path, 'w') as f:
            for img in image_files:
                f.write(f"{img}\n")
        print(f"{list_file_path} generated")

    def run_swarp(list_file_path, path_outim, path_swarp_conf, center_ra_hms, center_dec_dms, xsize=10200, ysize=6800, nthreads=8):
        """
        Run SWarp to stitch images together.
        
        Parameters:
        - list_file_path (str): File containing the list of masked image paths.
        - path_outim (str): Output filename for the mosaic image.
        - path_swarp_conf (str): Path to the SWarp configuration file.
        - center_ra_hms (str): Central RA in HMS format.
        - center_dec_dms (str): Central Dec in DMS format.
        - xsize (int): Final mosaic image width (in pixels).
        - ysize (int): Final mosaic image height (in pixels).
        - nthreads (int): Number of threads for SWarp.
        """
        # Construct dummy weight file name (not used for mask images)
        path_weightim = os.path.splitext(path_outim)[0] + '.weight.fits'
        
        swarp_command = (
            f'swarp @{list_file_path} -c {path_swarp_conf} '
            f'-IMAGEOUT_NAME {path_outim} '
            f'-WEIGHTOUT_NAME {path_weightim} '
            f'-CENTER "{center_ra_hms},{center_dec_dms}" '
            f'-IMAGE_SIZE {xsize},{ysize} '
            f'-NTHREADS {nthreads} '
        )
        print("Executing SWarp command for masked images:")
        print(swarp_command)
        print("=" * 60)
        os.system(swarp_command)
    
class PointingGenerator:
    
    def __init__(self):
        pass
    
    def generate_pointings(self, center_ra, center_dec, xsize, ysize, pixscale, n=9, m=6, margin_frac=0.1):
        """
        Generate RA, Dec pointings based on input center coordinates and image size.
        
        Parameters:
        - center_ra (float): Central RA coordinate in degrees.
        - center_dec (float): Central Dec coordinate in degrees.
        - xsize (int): X size of the image in pixels.
        - ysize (int): Y size of the image in pixels.
        - pixscale (float): Pixel scale in arcseconds/pixel.
        - n (int): Number of pointings along the X-axis.
        - m (int): Number of pointings along the Y-axis.
        - margin_frac (float): Fractional margin to add around the image.
        
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

    def plot_rectangle_on_sky(self, center_ra, center_dec, x_size, y_size, pixel_scale, color='blue', alpha=0.5):
        """
        Plot a rectangle on the sky with a given center and size in arcseconds.
        
        Parameters:
        - center_ra (float): Central RA in degrees.
        - center_dec (float): Central Dec in degrees.
        - x_size (int): X size of the image in pixels.
        - y_size (int): Y size of the image in pixels.
        - pixel_scale (float): Pixel scale in arcseconds per pixel.
        - color (str): Color of the rectangle.
        - alpha (float): Transparency of the rectangle.
        """
        # Convert size from pixels to degrees
        x_deg = x_size * pixel_scale / 3600.0
        y_deg = y_size * pixel_scale / 3600.0
        
        # Define rectangle edges
        ra_min = center_ra - x_deg / 2
        ra_max = center_ra + x_deg / 2
        dec_min = center_dec - y_deg / 2
        dec_max = center_dec + y_deg / 2

        # Plot the rectangle (as a series of lines between the corners)
        plt.plot([ra_min, ra_max], [dec_min, dec_min], color=color, alpha=alpha)  # Bottom edge
        plt.plot([ra_min, ra_max], [dec_max, dec_max], color=color, alpha=alpha)  # Top edge
        plt.plot([ra_min, ra_min], [dec_min, dec_max], color=color, alpha=alpha)  # Left edge
        plt.plot([ra_max, ra_max], [dec_min, dec_max], color=color, alpha=alpha)  # Right edge
        
    def plot_pointings_on_sky(self, pointings, center_ra, center_dec, x_size, y_size, pixel_scale, output_path, title=None, save=True):
        """
        Plot the pointings and the central rectangle on the sky.
        
        Parameters:
        - pointings (list of tuples): List of (RA, Dec) for each pointing.
        - center_ra (float): Central RA in degrees.
        - center_dec (float): Central Dec in degrees.
        - x_size (int): X size of the image in pixels.
        - y_size (int): Y size of the image in pixels.
        - pixel_scale (float): Pixel scale in arcseconds per pixel.
        - output_path (str): Path to save the plot image.
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

        # Plot each image boundary at the pointings
        skymapper_size = 0.17  # SkyMapper max image size in degrees
        for ra, dec in pointings:
            self.plot_rectangle_on_sky(ra, dec, skymapper_size * 3600 / pixel_scale, skymapper_size * 3600 / pixel_scale, pixel_scale, color='blue', alpha=0.3)

        # Plot the center and the 7DT image boundary
        plt.plot(center_ra, center_dec, 'rx', ms=10, label='7DT Center')
        self.plot_rectangle_on_sky(center_ra, center_dec, x_size, y_size, pixel_scale, color='red')

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
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        plt.show()
        plt.close('all')
        

class SkyMapperDownloader:
    
    def __init__(self):
        self.url = "https://api.skymapper.nci.org.au/public/siap/dr4/query"
        pass
    
    def get_sky_mapper_images(self, tra, tdec, size="0.17", filters="r", format="image/fits", intersect="OVERLAPS"):
        '''
        ---------------------------------------------------------------------------
        Function: get_sky_mapper_images
        Description:
          For each RA/Dec coordinate provided (as lists), query the SkyMapper SIAP API
          and collect all resulting images for that coordinate. The function prints 
          the number of images found for each coordinate, then returns a combined 
          Pandas DataFrame containing all query results (duplicates removed based on 
          'unique_image_id') and filtered to include only 'main' type images.
        
        Parameters:
          tra (list or array): List of RA values in degrees.
          tdec (list or array): List of Dec values in degrees.
          size (str): Size parameter for SkyMapper query (in degrees, e.g., "0.17").
          filters (str): Desired filter band (e.g., "r").
          format (str): Output format ("image/fits").
          intersect (str): INTERSECT option (e.g., "OVERLAPS").
        
        Returns:
          DataFrame: Combined table of image query results.
        ---------------------------------------------------------------------------
        '''
        results = []
        total_images = 0  # To accumulate total images found across all coordinates

        # Loop over each coordinate pair
        for ra, dec in zip(tra, tdec):
            # Format coordinates with high precision (12 decimal places)
            pos_str = f"{ra:.12f},{dec:.12f}"
            # Construct parameters for each query
            params = {
                "POS": pos_str,
                "SIZE": size,
                "BAND": filters,
                "FORMAT": format,
                "INTERSECT": intersect
            }
            try:
                response = requests.get(self.url, params=params, timeout=(5,10))
                if response.status_code == 200:
                    # Parse the VOTable response
                    votable = parse(io.BytesIO(response.content))
                    table = votable.get_first_table().to_table()
                    # Remove multidimensional columns
                    valid_columns = [name for name in table.colnames if len(table[name].shape) <= 1]
                    filtered_table = table[valid_columns]
                    df_coord = filtered_table.to_pandas()
                    count = len(df_coord)
                    total_images += count
                    print(f"Coordinate RA={pos_str}: found {count} images.")
                    # Append the data if available
                    if not df_coord.empty:
                        results.append(df_coord)
                else:
                    print(f"Error: Query at RA={pos_str} returned status code {response.status_code}")
            except Exception as e:
                print(f"Exception occurred for RA={pos_str}: {e}")
        
        # Combine all results and remove duplicates based on 'unique_image_id'
        if results:
            combined_df = pd.concat(results, ignore_index=True)
            # Filter to only include 'main' type images
            combined_df = combined_df[combined_df["image_type"] == "main"]
            print(f"Total images found across all coordinates: {total_images}")
            print(f"Total 'main' images after merging: {len(combined_df)}")
            return combined_df
        else:
            print("No images found for any coordinates.")
            return pd.DataFrame()  # Return empty DataFrame if no results
        
    def download_sky_mapper_images_for_pointings(self, df, path_slice, n_retry=3):
        '''
        ---------------------------------------------------------------------------
        Function: download_sky_mapper_images_for_pointings
        Description:
        Downloads SkyMapper images from the provided DataFrame.
        The function iterates over each row, downloads the image from the URL
        (column "get_image") and saves it using the original image name.

        Parameters:
        df (DataFrame): The image query results.
        path_slice (str): Directory to save downloaded images.
        n_retry (int): Number of retries for failed downloads.

        Returns:
        List of file paths to the downloaded images.
        ---------------------------------------------------------------------------
        '''
        image_files = []
        for idx, row in df.iterrows():
            # Get the download URL and image name
            fits_url = row["get_image"]
            image_name = row["image_name"].replace(" ", "_")
            os.makedirs(path_slice, exist_ok=True)  # Ensure the directory exists
            file_name = os.path.join(path_slice, f"{image_name}.fits")
            
            print(f"Downloading: {file_name} ...")
            start_time = time.time()
            # If file already exists and size > 0, skip download
            if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                print(f"Image already exists: {file_name}")
            else:
                n_try = 0
                query_success = False
                # Retry loop for download
                while n_try < n_retry and not query_success:
                    try:
                        response = requests.get(fits_url, timeout=(5, 10))
                        if response.status_code == 200:
                            with open(file_name, "wb") as file:
                                file.write(response.content)
                            query_success = True
                            elapsed_time = time.time() - start_time
                            print(f"Downloaded: {file_name} ({elapsed_time:.2f} sec)")
                        else:
                            print(f"Failed (status code {response.status_code}): {file_name}")
                            n_try += 1
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        n_try += 1
            image_files.append(file_name)
        return image_files
        
    def get_sky_mapper_mask_images(tra, tdec, size="0.17", filters="r", intersect="OVERLAPS"):
        '''
        ---------------------------------------------------------------------------
        Function: get_sky_mapper_mask_images
        Description:
        For each RA/Dec coordinate provided (as lists), query the SkyMapper SIAP API
        using FORMAT="image/fits" (to get a table containing the 'get_mask' field),
        and collect all resulting mask image entries for that coordinate.
        The function prints the number of mask entries found for each coordinate, then
        returns a combined Pandas DataFrame containing all query results.

        Parameters:
        tra (list or array): List of RA values in degrees.
        tdec (list or array): List of Dec values in degrees.
        size (str): Size parameter for SkyMapper query (in degrees, e.g., "0.17").
        filters (str): Desired filter band (e.g., "r").
        intersect (str): INTERSECT option (e.g., "OVERLAPS").

        Returns:
        DataFrame: Combined table of mask image query results.
        ---------------------------------------------------------------------------
        '''
        results = []
        total_entries = 0  # To accumulate total mask entries found across all coordinates

        # Loop over each coordinate pair
        for ra, dec in zip(tra, tdec):
            # Format coordinates with high precision (12 decimal places)
            pos_str = f"{ra:.12f},{dec:.12f}"
            # Construct parameters for each query using FORMAT="image/fits" to obtain mask info
            params = {
                "POS": pos_str,
                "SIZE": size,
                "BAND": filters,
                "FORMAT": "image/fits",  # Use FITS format to get table with 'get_mask' column
                "INTERSECT": intersect
            }
            try:
                response = requests.get(self.url, params=params, timeout=(5,10))
                if response.status_code == 200:
                    # Parse the VOTable response
                    votable = parse(io.BytesIO(response.content))
                    table = votable.get_first_table().to_table()
                    # Remove multidimensional columns
                    valid_columns = [name for name in table.colnames if len(table[name].shape) <= 1]
                    filtered_table = table[valid_columns]
                    df_coord = filtered_table.to_pandas()
                    count = len(df_coord)
                    total_entries += count
                    print(f"Coordinate RA={pos_str}: found {count} mask entries.")
                    # Append the data if available
                    if not df_coord.empty:
                        results.append(df_coord)
                else:
                    print(f"Error: Query at RA={pos_str} returned status code {response.status_code}")
            except Exception as e:
                print(f"Exception occurred for RA={pos_str}: {e}")
        
        if results:
            combined_df = pd.concat(results, ignore_index=True)
            # Do not filter by image_type here; we rely on the 'get_mask' field
            print(f"Total mask entries found across all coordinates: {total_entries}")
            print(f"Total entries in combined table: {len(combined_df)}")
            return combined_df
        else:
            print("No mask entries found for any coordinates.")
            return pd.DataFrame()

    def download_sky_mapper_mask_images_for_pointings(df, path_mask, n_retry=3):
        '''
        ---------------------------------------------------------------------------
        Function: download_sky_mapper_mask_images_for_pointings
        Description:
          Downloads SkyMapper mask images from the provided DataFrame.
          The function iterates over each row, downloads the mask image from the URL
          (from the "get_mask" column) and saves it with a '_mask' suffix in the filename.
        
        Parameters:
          df (DataFrame): The mask image query results.
          path_mask (str): Directory to save downloaded mask images.
          n_retry (int): Number of retries for failed downloads.
        
        Returns:
          List of file paths to the downloaded mask images.
        ---------------------------------------------------------------------------
        '''
        image_files = []
        for idx, row in df.iterrows():
            # Use the get_mask URL for downloading the mask image
            mask_url = row["get_mask"]
            image_name = row["image_name"].replace(" ", "_")
            file_name = os.path.join(path_mask, f"{image_name}_mask.fits")
            
            print(f"Downloading mask: {file_name} ...")
            start_time = time.time()
            if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                print(f"Mask image already exists: {file_name}")
            else:
                n_try = 0
                query_success = False
                while n_try < n_retry and not query_success:
                    try:
                        response = requests.get(mask_url, timeout=(5,10))
                        if response.status_code == 200:
                            with open(file_name, "wb") as file:
                                file.write(response.content)
                            query_success = True
                            elapsed_time = time.time() - start_time
                            print(f"Downloaded mask: {file_name} ({elapsed_time:.2f} sec)")
                        else:
                            print(f"Failed (status code {response.status_code}): {file_name}")
                            n_try += 1
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        n_try += 1
            image_files.append(file_name)
        return image_files
    
    def match_science_and_mask(science_dir, mask_dir):
        """
        Check matching between science images and mask images based on file names.
        Assumes that the mask image file name is the same as the science image file name
        with an added "_mask" suffix before the extension.
        
        Parameters:
        - science_dir (str): Directory containing science FITS images.
        - mask_dir (str): Directory containing mask FITS images.
        
        Returns:
        - matched (list of tuples): List of (science_file, mask_file) pairs that match.
        - unmatched_science (list): List of science image files with no matching mask.
        - unmatched_mask (list): List of mask image files with no matching science image.
        """
        # List all FITS files in both directories
        science_files = sorted([f for f in os.listdir(science_dir) if f.endswith(".fits")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".fits")])
        
        # Create dictionaries mapping base names to file names.
        # For science images, base name is the file name without extension.
        science_dict = {os.path.splitext(f)[0]: f for f in science_files}
        
        # For mask images, remove the trailing '_mask' if present to get the base name.
        mask_dict = {}
        for f in mask_files:
            base = os.path.splitext(f)[0]
            if base.endswith("_mask"):
                base = base[:-5]
            mask_dict[base] = f
        
        matched = []
        unmatched_science = []
        unmatched_mask = []
        
        # Check for matching pairs
        for base, sci_file in science_dict.items():
            if base in mask_dict:
                matched.append((sci_file, mask_dict[base]))
            else:
                unmatched_science.append(sci_file)
        
        # Check mask images that do not have a corresponding science image
        for base, mask_file in mask_dict.items():
            if base not in science_dict:
                unmatched_mask.append(mask_file)
        
        return matched, unmatched_science, unmatched_mask

    def apply_mask_to_science_image(science_path, mask_path, output_path):
        """
        Load the science image and its corresponding mask image, create a binary mask
        where pixels with mask value == 0 (good pixels) are set to 1 and pixels with 
        mask value != 0 (bad pixels) are set to 0. Apply this binary mask to the science 
        image (pixel-wise multiplication) and save the masked image.
        
        Parameters:
        - science_path (str): Path to the science image FITS file.
        - mask_path (str): Path to the corresponding mask FITS file.
        - output_path (str): Path to save the masked science image.
        """
        # Open science image and get data and header
        with fits.open(science_path) as hdul_sci:
            sci_data = hdul_sci[0].data
            header = hdul_sci[0].header
        
        # Open mask image and get data
        with fits.open(mask_path) as hdul_mask:
            mask_data = hdul_mask[0].data
        
        # Check if the science image and mask image have the same dimensions
        if sci_data.shape != mask_data.shape:
            raise ValueError(f"Shape mismatch: {science_path} and {mask_path}")
        
        # Create binary mask: good pixels (mask==0) become 1, bad pixels (mask != 0) become 0
        binary_mask = np.where(mask_data == 0, 1, 0).astype(np.uint8)
        
        # Apply the binary mask to the science image
        masked_data = sci_data * binary_mask
        
        # Save the masked image with the original header (to preserve WCS and other metadata)
        hdu = fits.PrimaryHDU(masked_data, header=header)
        hdu.writeto(output_path, overwrite=True)
        print(f"Masked science image saved to {output_path}")
# %%
if __name__ == "__main__":
    from tippy.utils import Tiles
    filterset = 'gri'
    T = Tiles()
    SMPointing = PointingGenerator()
    SMStitch = SkyMapperStitch()
    SMDownloader = SkyMapperDownloader()
    tile_id_list = list([
    "T01259", "T01159", "T01467", "T01361", "T01160", "T01360", "T01258", "T01362",
    "T01256", "T01363", "T01161", "T01257", "T01469", "T01260", "T01468", "T01158",
    "T01465", "T01576", "T01064", "T01358", "T01359", "T01364", "T01065", "T01261",
    "T01466", "T01162", "T01066", "T01463", "T01577", "T00974", "T01063", "T01578",
    "T01067", "T01262", "T01470", "T01691", "T01255", "T01357", "T00975", "T01575",
    "T00973", "T01163", "T01157", "T01579", "T01690", "T01692", "T01156", "T01464",
    "T01689", "T00972", "T01693", "T01580", "T01462", "T01574", "T01068", "T00976",
    "T01062", "T01573", "T01254", "T01365", "T01694", "T01572", "T01355", "T01164",
    "T01471", "T01809", "T01356", "T01810", "T01263", "T01811", "T01061", "T01688",
    "T01687", "T01461", "T01155"
    ])
    for tile_id in tile_id_list:
        tile_info = T.get_tile_info(tile_id)
        ra = tile_info['ra'][0]
        dec = tile_info['dec'][0]
        x_size = 12000
        y_size = 8000
        pixelscale = 0.505
        n = 9
        m = 6
        pointings = SMPointing.generate_pointings(center_ra = ra,
                                      center_dec = dec,
                                      xsize = x_size,
                                      ysize = y_size,
                                      pixscale = pixelscale,
                                      n = n,
                                      m = m,
                                      margin_frac = 0.1)
        print(f'{len(pointings)} pointings generated for {tile_id} at RA={ra}, Dec={dec}')
        # SMPointing.plot_pointings_on_sky(
        #     pointings = pointings,
        #     center_ra = ra,
        #     center_dec = dec,
        #     x_size = x_size,
        #     y_size = y_size,
        #     pixel_scale = pixelscale,
        #     output_path = f'/home/hhchoi1022/data/refdata/main/Skymapper/Skymapper_DR1/T01462/SMSS/{tile_id}_pointings.png',
        #     title = f'SkyMapper Pointings for {tile_id}',
        #     save = True
        # )
        # Query Skymapper 
        for filter_ in filterset:
            tra = [p[0] for p in pointings]
            tdec = [p[1] for p in pointings]
            result = SMDownloader.get_sky_mapper_images(
                tra = tra,
                tdec = tdec,
                size = "0.17",
                filters = filter_,
                format = "image/fits",
                intersect = "OVERLAPS"
            )
            
            SMDownloader.download_sky_mapper_images_for_pointings(
                result,
                path_slice = f'/home/hhchoi1022/data/refdata/main/Skymapper/Skymapper_DR1/{tile_id}/SMSS/{filter_}'
            )
            
            time.sleep(60)
        