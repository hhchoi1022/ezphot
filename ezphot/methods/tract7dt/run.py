
#%%

import subprocess
from datetime import datetime
import os
import glob
from typing import List, Optional, Union
from pathlib import Path
from astropy.io import ascii
from astropy.table import Table
from ezphot.methods.tract7dt import Formatter
from ezphot.methods.tract7dt import Configuration
from ezphot.utils.tiles import Tiles
#%%

class Tract7DTRunner:
    def __init__(self,
                 image_paths: List[Path],
                 filter_list: List[str],
                 catalog_paths: Optional[List[Path]] = None,
                 conda_exe = f"{Path.home()}/anaconda3/condabin/conda",
                 conda_env: str = 'tractor',
                 python_exe: str = 'python',
                 id: str = None,
                 base_dir: Path = Path.home() / 'tract7dt'):
        self.conda_exe = conda_exe
        self.conda_env = conda_env
        self.python_exe = python_exe
        self.image_paths = image_paths
        self.filter_list = filter_list
        self.catalog_paths = catalog_paths
        self.formatter = Formatter(image_paths = image_paths,
                                   filter_list = filter_list,
                                   catalog_paths = catalog_paths)
        self.configuration = Configuration()
        if id is None:
            self.id = datetime.now().strftime("%Y%m%d_%H%M")
        else:
            self.id = id
        print(f"ID: {self.id}")
        self.workdir = Path(base_dir) / self.id
        print('Tract7DT working directory: ', self.workdir)
        self._register_configuration_paths()
        
    def __repr__(self):
        return f"Tract7DTRunner(id = {self.id}, workdir = {self.workdir}, image_list_saved = {self.input_images_saved}, catalog_list_saved = {self.input_catalogs_saved})"
    
    
    def _register_configuration_paths(self):
        self.configuration.inputs.input_catalog = self.input_catalogs_path
        self.configuration.inputs.image_list_file = self.input_images_path
        self.configuration.outputs.work_dir = self.workdir    
        
    @property
    def input_images_saved(self):
        return self.input_images_path.exists()
    
    @property
    def input_catalogs_saved(self):
        return self.input_catalogs_path.exists()
    
    @property
    def reference_catalog_saved(self):
        return Path(self.configuration.inputs.gaiaxp_synphot_csv).exists()
    
    @property
    def input_images_path(self):
        return self.workdir / f"input_images.txt"

    @property
    def input_catalogs_path(self):
        return self.workdir / f"input_catalogs.csv"
    
    @property
    def configuration_path(self):
        return self.workdir / f"configuration.yaml"
        
    def register_target(self, 
                        list_ra: List[float],
                        list_dec: List[float],
                        list_type: List[str] = None,
                        list_flux: List[float] = None,
                        list_ellip: List[float] = None,
                        list_Re: List[float] = None,
                        list_theta: List[float] = None,
                        
                        update_type_from_catalog: bool = True,
                        update_flux_from_catalog: bool = True,
                        update_ellip_from_catalog: bool = True,
                        update_Re_from_catalog: bool = True,
                        update_theta_from_catalog: bool = True,
                        
                        ra_key: str = 'X_WORLD',
                        dec_key: str = 'Y_WORLD',
                        type_key: str = 'CLASS_STAR',
                        ellip_key: str = 'ELLIPTICITY',
                        flux_key: str = 'FLUX_AUTO',
                        Re_key: str = 'FLUX_RADIUS',
                        theta_key: str = 'THETA_WORLD',
                        matching_radius_arcsec: float = 5):
        
        self.input_images_path.parent.mkdir(parents=True, exist_ok=True)
        self.input_catalogs_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.formatter.to_input_images(output_path = self.input_images_path)
        
        self.formatter.to_input_catalogs(output_path = self.input_catalogs_path,
                                         list_ra = list_ra,
                                         list_dec = list_dec,
                                         list_flux = list_flux,
                                         list_type = list_type,
                                         list_ellip = list_ellip,
                                         list_Re = list_Re,
                                         list_theta = list_theta,
                                         update_type_from_catalog = update_type_from_catalog,
                                         update_flux_from_catalog = update_flux_from_catalog,
                                         update_ellip_from_catalog = update_ellip_from_catalog,
                                         update_Re_from_catalog = update_Re_from_catalog,
                                         update_theta_from_catalog = update_theta_from_catalog,
                                         ra_key = ra_key,
                                         dec_key = dec_key,
                                         type_key = type_key,
                                         ellip_key = ellip_key,
                                         flux_key = flux_key,
                                         Re_key = Re_key,
                                         theta_key = theta_key,
                                         matching_radius_arcsec = matching_radius_arcsec)
        
        print('Input images and catalogs saved to: ', self.input_images_path, self.input_catalogs_path)
        
    def register_reference_catalog(self, 
                                   catalog_path: Union[Path, str] = None,
                                   objname: str = None,
                                   list_ra: List[float] = None,
                                   list_dec: List[float] = None,
                                   gaiaxp_catalog_dir: Union[Path, str] = '/lyman/data1/factory/ref_cat'):
        reference_catalog_path = None
        if catalog_path is not None:
            if Path(catalog_path).exists():
                reference_catalog_path = str(catalog_path)
        if objname is not None:
            catalog_path = glob.glob(os.path.join(gaiaxp_catalog_dir, f'*{objname}*'))[0]
            if Path(catalog_path).exists():
                reference_catalog_path = str(catalog_path)
        if list_ra is not None and list_dec is not None:
            catalog_path = self.find_gaiaxp_catalog(list_ra = list_ra, list_dec = list_dec, gaiaxp_catalog_dir = gaiaxp_catalog_dir)
            if Path(catalog_path).exists():
                reference_catalog_path = str(catalog_path)
        
        if reference_catalog_path is None:
            raise RuntimeError('No reference catalog provided. Please provide a catalog path or objname or list_ra and list_dec.')
        self.configuration.inputs.gaiaxp_synphot_csv = reference_catalog_path                                   
        
    def find_gaiaxp_catalog(self, 
                            list_ra: List[float],
                            list_dec: List[float],
                            tileinfo_path: Union[Path, str] = f'{Path.home()}/ezphot/data/tileinfo/7-DT/final_tiles.txt',
                            gaiaxp_catalog_dir: Union[Path, str] = '/lyman/data1/factory/ref_cat'):
        tiles = Tiles(tileinfo_path)
        matched_tile_tbl, matched_coords_dict, _ = tiles.find_overlapping_tiles(list_ra = list_ra, list_dec = list_dec, visualize = False)

        len_per_tile = dict()
        for tile_id, matched_coords in matched_coords_dict.items():
            len_per_tile[tile_id] = len(matched_coords)
        most_probable_tile = max(len_per_tile, key=len_per_tile.get)
        
        gaiaxp_catalog_path = glob.glob(os.path.join(gaiaxp_catalog_dir, f'*{most_probable_tile}*'))[0]
        return gaiaxp_catalog_path
    
    def run(self, save_result_catalog: bool = True):
        original_dir = os.getcwd()
        os.chdir(self.workdir)
        self.configuration.to_yaml(path = self.configuration_path)
        
        # Check whether input_images and input_catalogs are saved
        if not self.input_images_saved or not self.input_catalogs_saved:
            raise ValueError('Input images and catalogs are not saved. Please register targets first. Run self.register_target() first.')
        if not self.reference_catalog_saved:
            raise ValueError('Reference catalog is not saved. Please register reference catalog first. Run self.register_reference_catalog() first.')
        
        # Use subprocess
        command = [
            self.conda_exe,
            "run",
            "-n",
            self.conda_env,
            "tract7dt",
            "run",
            "--config",
            str(self.configuration_path),
        ]

        print('Running tract7dt...')
        try:
            subprocess.run(command, check=True)
            if save_result_catalog:
                self.save_result_catalog()
        except subprocess.CalledProcessError as e:
            print(f"Error during tract7dt execution: {e.stderr.decode()}")
            raise e
        finally:
            os.chdir(original_dir)
            
    def save_result_catalog(self):
        kwargs_map = {}
        kwargs_map['RA_fit'] = 'X_WORLD'
        kwargs_map['DEC_fit'] = 'Y_WORLD'
        kwargs_map['patch_tag'] = 'PATCH_TAG'
        kwargs_map['epsf_tag'] = 'EPSF_TAG'
        kwargs_map['psf_used_epsf_band_count'] = 'N_PSFSTAR'
        kwargs_map['opt_niters'] = 'NITERS'
        kwargs_map['opt_converged'] = 'CONVERGED'
        kwargs_map['stype_fit'] = 'TYPE'
        kwargs_map['sersic_n_fit'] = 'SERSIC_IDX'
        kwargs_map['ab_fit'] = 'AB'
        kwargs_map['phi_deg_fit'] = 'THETA_WORLD'
        kwargs_map['ELL_fit'] = 'ELLIPTICITY'
        kwargs_map['Re_fit'] = 'Re'
        kwargs_map['THETA_fit'] = 'THETA_IMAGE'

        result_catalog_path = self.workdir / 'final_catalog_with_fit.csv'
        zp_path = self.workdir / 'ZP' / 'zp_summary.csv'
        if not result_catalog_path.exists():
            print('Catalog is not saved. Please run tract7dt first.')
            return
        if zp_path.exists():
            zp_info_tbl = ascii.read(zp_path)
        else:
            zp_info_tbl = Table()
            
        result_tbl = ascii.read(result_catalog_path)
        for target_path, target_filter in zip(self.image_paths, self.filter_list):
            savedir = Path(target_path).parent
            filename = Path(target_path).name
            catalogpath = savedir / (filename + '.tract7dtcat')
            
            flux_key_fit = f'FLUX_{target_filter}_fit'
            fluxerr_key_fit = f'FLUXERR_{target_filter}_fit'
            mag_key_fit = f'MAG_{target_filter}_fit'
            magerr_key_fit = f'MAGERR_{target_filter}_fit'
            
            catalog_tbl = Table()
            for key_fit, key_cat in kwargs_map.items():
                catalog_tbl[key_cat] = result_tbl[key_fit]
                
            catalog_tbl['FLUX_TRACT7DT'] = result_tbl[flux_key_fit]
            catalog_tbl['FLUXERR_TRACT7DT'] = result_tbl[fluxerr_key_fit]
            catalog_tbl['MAG_TRACT7DT'] = result_tbl[mag_key_fit]
            catalog_tbl['MAGERR_TRACT7DT'] = result_tbl[magerr_key_fit]
            
            zp_row = zp_info_tbl[zp_info_tbl['band'] == target_filter]
            if len(zp_row) == 1:
                zp = zp_row['zp_median']
                zperr = zp_row['zp_err_mad']
            else:
                zp = None
                zperr = None
            
            if zp is not None:
                catalog_tbl['ZP_TRACT7DT'] = zp
                catalog_tbl['ZPERR_TRACT7DT'] = zperr
            
            catalog_tbl.write(catalogpath, format = 'ascii', overwrite = True)
            print(f'Catalog saved to {catalogpath}')        
# %%

# %%