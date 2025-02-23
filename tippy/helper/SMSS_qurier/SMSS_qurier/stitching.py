import os

def write_images_to_swarp(list_file_path, image_files):
    # Create the file list for SWarp input
    # list_file_path = f"{os.path.dirname(output_file)}/images_to_combine.txt"
    with open(list_file_path, 'w') as f:
        for img in image_files:
            f.write(f"{img}\n")
    print(f"{list_file_path} generated")

def run_swarp(list_file_path, path_outim, path_swarp_conf, center_ra_hms, center_dec_dms, xsize=10200, ysize=6800, nthreads=8):
    """Run SWarp to stitch images together."""
    # Example: Construct the system call to swarp
    path_weightim = os.path.splitext(path_outim)[0] + '.weight.fits'

    swarp_command = (
        f'swarp @{list_file_path} -c {path_swarp_conf} '
        f'-IMAGEOUT_NAME {path_outim} '
        f'-WEIGHTOUT_NAME {path_weightim} '
        f'-CENTER "{center_ra_hms},{center_dec_dms}" '
        f'-IMAGE_SIZE {xsize},{ysize} '  # 7DT image size
        f'-NTHREADS {nthreads} '
    )
    print(swarp_command)
    print(f"="*60)
    os.system(swarp_command)