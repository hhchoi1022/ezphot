#%%
from pathlib import Path
import json
from typing import Union
import shutil
import numpy as np

class Configuration:
    def __init__(self,
                 telkey: str = None,
                 configpath: Union[Path, str] = Path.home()/'ezphot/config'):#Path(__file__).resolve().parent):
        if not Path(configpath).exists():
            print(f"[CRITICAL] Configuration path {configpath} does not exist. Run initialization first.")
        self.telkey = telkey
        self.config = dict()

        # Set up paths
        self.path_home = Path.home()
        self.path_base = Path(__file__).resolve().parent.parent
        self.path_config = Path(configpath)
        self.path_ezphot =  self.path_config.parent
        self.path_log = self.path_ezphot / 'log'
        self.path_data = self.path_ezphot / 'data'
        
        # Gloabl configuration files
        self.path_config_global = self.path_config / 'common'
        self._configfiles_global = list(self.path_config_global.glob('*.config'))
        config_global = self._load_configuration(self._configfiles_global)
        self.config.update(config_global)
        
        # Telescope specific configuration files
        self.path_config_specific = self.path_config / 'specific'
        if self.telkey is not None:
            self.telinfo = self._load_telinfo()
            self.path_config_specific_telescope = self.path_config_specific / self.telkey
            self._configfiles_telescopes = list(self.path_config_specific_telescope.glob('*.config'))
            if not self._configfiles_telescopes:
                print('No configuration file is found.\n Do you want to make default configuration files? (y/n)')
                answer = input()
                if answer == 'y':
                    self.register_telescope()
                    self._configfiles_telescopes = list(self.path_config_specific_telescope.glob('*.config'))
                    config_unit = self._load_configuration(self._configfiles_telescopes)
                    self.config.update(config_unit)    
                else:
                    print("WARNING: Only common configuration files are loaded.")
            else:
                config_unit = self._load_configuration(self._configfiles_telescopes)
                self.config.update(config_unit)           
                
    def initialize(self, copy_default: bool = True):
        """Initialize the configuration by creating necessary config files."""
        # Make sure base paths exist
        self.path_config_global.mkdir(parents=True, exist_ok=True)
        self.path_config_specific.mkdir(parents=True, exist_ok=True)
        self.path_log.mkdir(parents=True, exist_ok=True)
        self.path_data.mkdir(parents=True, exist_ok=True)

        print(f"Global configuration path created: {self.path_config_global}")
        print(f"Specific configuration path created: {self.path_config_specific}")
        
        default_global_config_path = self.path_base / 'configuration' /  'common'
        default_specific_config_path = self.path_base / 'configuration' / 'specific'
        # Copy default config files to the folder
        if copy_default:
            shutil.copytree(default_global_config_path, self.path_config_global, dirs_exist_ok=True)
            print(f"Copied default global configs from {default_global_config_path}")
            shutil.copytree(default_specific_config_path, self.path_config_specific, dirs_exist_ok=True)
            print(f"Copied default telescope configs from {default_specific_config_path}")
            telescope_keys = ['\n' + p.name for p in default_specific_config_path.iterdir() if p.is_dir()]
            telescope_keys_str = ' '.join(telescope_keys)
            print(f'Current available telescope keys: {telescope_keys_str}')
            
        for tel_key in self.available_telescope_keys:
            # Rsgister each telescope
            instance = Configuration(telkey=tel_key)
            instance.path_config_specific_telescope.mkdir(parents=True, exist_ok=True)
            instance._register_telescope()
        
        self = Configuration()
        # After creating all config files, load them and make sure all directories exist
        for path in list(self.config.values()):
            self._ensure_dirs_exist(path)
            
    def register_telescope(self):
        if not self.telkey:
            raise ValueError("Telescope key (telkey) must be provided to initialize configuration.")

        self.path_config_specific_telescope.mkdir(parents=True, exist_ok=True)

        # Create config files
        self._register_telescope()
        
        for path in list(self.config.values()):
            self._ensure_dirs_exist(path)
            
    @property
    def available_telescope_keys(self):
        """Return a list of available telescope keys."""
        return [p.name for p in self.path_config_specific.iterdir() if p.is_dir()]
    
    def _load_telinfo(self):
        from astropy.io import ascii
        telinfo_tbl = ascii.read(self.path_config_global / 'observatory_info.dat', format='fixed_width')
        
        # Parse telescope key
        telkey_keys = self.telkey.split('_')
        if len(telkey_keys) == 4:
            observatory, ccd, readoutmode, binning_str = telkey_keys
        elif len(telkey_keys) == 3:
            observatory, ccd, binning_str = telkey_keys
            readoutmode = None
        else:
            raise ValueError(f"Invalid telescope key: {self.telkey}. Telescope key should be in the format of 'observatory_ccd_readoutmode_binning' or 'observatory_ccd_binning'")
        
        # Extract binning number from format like "1x1" -> 1
        binning = int(binning_str.split('x')[0])
        
        # Filter by observatory
        telinfo = telinfo_tbl[telinfo_tbl['telescope'] == observatory]
        
        # Filter by CCD
        telinfo = telinfo[telinfo['ccd'] == ccd]
        
        # Filter by binning
        telinfo = telinfo[telinfo['binning'] == binning]
        
        # Filter by readout mode if provided
        if readoutmode is not None:
            telinfo = telinfo[telinfo['readoutmode'] == readoutmode]
        
        # Return result
        if len(telinfo) == 0:
            raise ValueError(f"No telescope information found for {self.telkey} in observatory_info.dat. \n Please follow the below instructions. \n 1. Enter necessary telescope information to {self.path_config_global / 'observatory_info.dat'} \n 2. Register by running Configuration(telkey).register_telescope()")
        else:
            return telinfo[0]
        
    def _load_configuration(self, configfiles):
        all_config = dict()
        for configfile in configfiles:
            with open(configfile, 'r') as f:
                config = json.load(f)
                all_config.update(config)
        return all_config

    def _make_configfile(self, dict_params: dict, filename: str, savepath: Union[str, Path]):
        filepath = Path(savepath) / filename
        with open(filepath, 'w') as f:
            json.dump(dict_params, f, indent=4)
        print(f'New configuration file made: {filepath}')

    def _ensure_dirs_exist(self, *paths):
        """Create all directories in the given paths if they don't exist."""
        for p in paths:
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                pass
                #print(f"[WARNING] Could not create directory {p}: {e}")
                
    def _register_telescope(self):
        
        # Specific telescope configuration
        sex_config = dict(
            SEX_CONFIG = str(self.path_config_specific_telescope / f'{self.telkey}.sexconfig'),
            SEX_DIR = str(self.path_config_global / 'sextractor'),
            SEX_LOGDIR = str(self.path_log / 'sextractor' / 'log'),
            SEX_HISTORYDIR = str(self.path_log / 'sextractor' / 'history')
        )
        astrometry_config = dict(
            ASTROMETRY_SEXCONFIG = str(self.path_config_specific_telescope / f'{self.telkey}.astrometry.sexconfig')
        )
        scamp_config = dict(
            SCAMP_CONFIG = str(self.path_config_specific_telescope / f'{self.telkey}.scampconfig'),
            SCAMP_SEXCONFIG = str(self.path_config_specific_telescope / f'{self.telkey}.scamp.sexconfig'),
            SCAMP_DIR = str(self.path_config_global / 'scamp'),
            SCAMP_LOGDIR = str(self.path_log / 'scamp' / 'log'),
            SCAMP_HISTORYDIR = str(self.path_log / 'scamp' / 'history')
        )
        swarp_config = dict(
            SWARP_CONFIG = str(self.path_config_specific_telescope / f'{self.telkey}.swarpconfig'),
            SWARP_DIR = str(self.path_config_global / 'swarp'),
            SWARP_LOGDIR = str(self.path_log / 'swarp' / 'log'),
            SWARP_HISTORYDIR = str(self.path_log / 'swarp' / 'history')
        )
        psfex_config = dict(
            PSFEX_CONFIG = str(self.path_config_specific_telescope / f'{self.telkey}.psfexconfig'),
            PSFEX_SEXCONFIG = str(self.path_config_specific_telescope / f'{self.telkey}.psfex.sexconfig'),
            PSFEX_DIR = str(self.path_config_global / 'psfex'),
            PSFEX_LOGDIR = str(self.path_log / 'psfex' / 'log'),
            PSFEX_HISTORYDIR = str(self.path_log / 'psfex' / 'history')
        )
        
        for cfg, name in [
            (sex_config, 'sex.config'),
            (astrometry_config, 'astrometry.config'),
            (scamp_config, 'scamp.config'),
            (swarp_config, 'swarp.config'),
            (psfex_config, 'psfex.config')
        ]:
            self._make_configfile(cfg, name, self.path_config_specific_telescope)
        
        # Update default configuration files to the telescope specific ones
        default_sexconfig = self.path_config_global / 'default.sexconfig'
        default_astrometrysexconfig = self.path_config_global / 'default.astrometry.sexconfig'
        default_swarpconfig = self.path_config_global / 'default.swarpconfig'
        default_scampconfig = self.path_config_global / 'default.scampconfig'
        default_scampsexconfig = self.path_config_global / 'default.scamp.sexconfig'
        default_psfexconfig = self.path_config_global / 'default.psfexconfig'
        default_psfexsexconfig = self.path_config_global / 'default.psfex.sexconfig'
        self._update_default_sexconfig(default_sexconfig, sex_config['SEX_CONFIG'])
        self._update_default_sexconfig(default_astrometrysexconfig, astrometry_config['ASTROMETRY_SEXCONFIG'])
        self._update_default_swarpconfig(default_swarpconfig, swarp_config['SWARP_CONFIG'])
        self._update_default_scampconfig(default_scampconfig, scamp_config['SCAMP_CONFIG'])
        self._update_default_sexconfig(default_scampsexconfig, scamp_config['SCAMP_SEXCONFIG'])
        self._update_default_psfexconfig(default_psfexconfig, psfex_config['PSFEX_CONFIG'])
        self._update_default_sexconfig(default_psfexsexconfig, psfex_config['PSFEX_SEXCONFIG'])

        # Global configuration
        calibdata_config = dict(
            CALIBDATA_DIR = str(self.path_data / 'calibdata'),
            CALIBDATA_MASTERDIR = str(self.path_data / 'mcalibdata'),
            )
        refdata_config = dict(
            REFDATA_DIR = str(self.path_data / 'refdata'),
            )
        obsdata_config = dict(OBSDATA_DIR = str(self.path_data / 'obsdata'))
        scidata_config = dict(SCIDATA_DIR = str(self.path_data / 'scidata'))
        catalog_config = dict(CATALOG_DIR = str(self.path_data / 'skycatalog' / 'archive'))
        observatory_config = dict(
            OBSERVATORY_LOCATIONINFO = str(self.path_config_global / 'obs_location.txt'),
            OBSERVATORY_TELESCOPEINFO = str(self.path_config_global / 'observatory_info.dat'),
            OBSERVATORY_TELESCOPEHINT = str(self.path_config_global / 'observatory_info_hint.yaml')
        )
        
        sdtdata_config = dict(
            SDTDATA_OBSSOURCEDIR = str(self.path_data / 'connecteddata' / '7DT' / 'obsdata'),
            SDTDATA_OBSDESTDIR = str(self.path_data / 'obsdata' / '7DT'),
            SDTDATA_SCISOURCEDIR = str(self.path_data / 'connecteddata' / '7DT' / 'processed_1x1_gain2750'),
            SDTDATA_SCIDESTDIR = str(self.path_data / 'scidata' / '7DT' / '7DT_C361K_HIGH_1x1')
        )

        for cfg, name in [
            (calibdata_config, 'calibdata.config'),
            (refdata_config, 'refdata.config'),
            (scidata_config, 'scidata.config'),
            (catalog_config, 'catalog.config'),
            (observatory_config, 'observatory.config'),
            (sdtdata_config, 'sdtdata.config'),
            (obsdata_config, 'obsdata.config')
        ]:
            self._make_configfile(cfg, name, self.path_config_global)

        # Remove per-telescope specific keys before saving global versions
        sex_config['SEX_CONFIG'] = str(default_sexconfig)
        scamp_config['SCAMP_CONFIG'] = str(default_scampconfig)
        scamp_config['SCAMP_SEXCONFIG'] = str(default_scampsexconfig)
        swarp_config['SWARP_CONFIG'] = str(default_swarpconfig)
        psfex_config['PSFEX_CONFIG'] = str(default_psfexconfig)
        psfex_config['PSFEX_SEXCONFIG'] = str(default_psfexsexconfig)
        astrometry_config['ASTROMETRY_SEXCONFIG'] = str(default_astrometrysexconfig)

        for cfg, name in [
            (sex_config, 'sex.config'),
            (scamp_config, 'scamp.config'),
            (swarp_config, 'swarp.config'),
            (psfex_config, 'psfex.config'),
            (astrometry_config, 'astrometry.config'),
        ]:
            self._make_configfile(cfg, name, self.path_config_global)
    
    def _update_default_sexconfig(self, default_path: Union[str, Path] = None, output_path: Union[str, Path] = None):
        
        telinfo = self.telinfo
        if default_path is None:
            default_path = self.path_config_global / 'default.sexconfig'
        if output_path is None:
            output_path = self.path_config_specific_telescope / f'{self.telkey}.sexconfig'
        # Define updates based on telescope info
        update_dict = {}
        if telinfo is not None:
            # Update based on telescope properties
            update_dict['GAIN'] = telinfo['gain']
            update_dict['READNOISE'] = telinfo['readnoise']
            update_dict['PIXEL_SCALE'] = telinfo['pixelscale']
            update_dict['DETECT_MINAREA'] = int(np.pi* (2/telinfo['pixelscale']/2)**2) if telinfo['pixelscale'] < 1 else int(np.pi* (4/telinfo['pixelscale']/2)**2)
            update_dict['SEEING_FWHM'] = 2.5 if telinfo['pixelscale'] < 1 else 4
            update_dict['PHOT_APERTURES'] = f'{round(5 / telinfo["pixelscale"],2)},{round(7 / telinfo["pixelscale"],2)},{round(10 / telinfo["pixelscale"],2)}'
            if update_dict['DETECT_MINAREA'] < 3:
                update_dict['DETECT_MINAREA'] = 3

        self._update_defulatconfig(update_dict, default_path, output_path)

    def _update_default_swarpconfig(self, default_path: Union[str, Path] = None, output_path: Union[str, Path] = None):
        
        telinfo = self.telinfo
        if default_path is None:
            default_path = self.path_config_global / 'default.swarpconfig'
        if output_path is None:
            output_path = self.path_config_specific_telescope / f'{self.telkey}.swarpconfig'
        # Define updates based on telescope info
        update_dict = {}
        if telinfo is not None:
            # Update based on telescope properties
            update_dict['GAIN_DEFAULT'] = telinfo['gain']
            update_dict['PIXEL_SCALE'] = telinfo['pixelscale']
            update_dict['IMAGE_SIZE'] = f'{round(1.1*telinfo["x"],-2)},{round(1.1*telinfo["y"],-2)}'

        self._update_defulatconfig(update_dict, default_path, output_path)

    def _update_default_scampconfig(self, default_path: Union[str, Path] = None, output_path: Union[str, Path] = None):
        
        if default_path is None:
            default_path = self.path_config_global / 'default.scampconfig'
        if output_path is None:
            output_path = self.path_config_specific_telescope / f'{self.telkey}.scampconfig'
        
        update_dict = dict()
        self._update_defulatconfig(update_dict, default_path, output_path)
        
    def _update_default_psfexconfig(self, default_path: Union[str, Path] = None, output_path: Union[str, Path] = None):
        
        if default_path is None:
            default_path = self.path_config_global / 'default.psfexconfig'
        if output_path is None:
            output_path = self.path_config_specific_telescope / f'{self.telkey}.psfexconfig'
            
        update_dict = dict()
        self._update_defulatconfig(update_dict, default_path, output_path)
    
    def _update_defulatconfig(self, 
                              update_dict: dict, 
                              default_path: Union[str, Path], 
                              output_path: Union[str, Path]):

        """Update specific keywords in the default.sexconfig based on telescope info."""
        
        # Read the original file
        config_path = default_path
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Update the lines
        updated_lines = []
        for line in lines:
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                updated_lines.append(line)
                continue
            
            # Check if this line contains a key we want to update
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                if key in update_dict:
                    # Replace the value while preserving formatting
                    value = update_dict[key]
                    # Keep the original spacing and comments
                    comment_start = line.find('#')
                    if comment_start != -1:
                        comment = line[comment_start:]
                        # Format: KEY    VALUE    # comment
                        updated_line = f"{key:<15} {value:<10} {comment}\n"
                    else:
                        updated_line = f"{key:<15} {value}\n"
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
                        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write the updated configuration
        with open(output_path, 'w') as f:
            f.writelines(updated_lines)
        
        print("New configuration file made: ", output_path)

#%%
if __name__ == '__main__':
    # For the first time, initiailize the configuration
    self = Configuration()
    #self.initialize(copy_default=True)
    
    telescope_keys = [
        '7DT_C361K_HIGH_1x1', '7DT_C361K_HIGH_2x2', '7DT_C361K_LOW_1x1', '7DT_C361K_LOW_2x2',
        'CBNUO_STX16803_1x1', 'LSGT_SNUCAMII_1x1', 'LSGT_ASI1600MM_1x1',
        'RASA36_KL4040_HIGH_1x1', 'RASA36_KL4040_MERGE_1x1', 'SAO_C361K_1x1',
        'SOAO_FLI4K_1x1', 'KCT_STX16803_1x1']

    #self = Configuration(telkey='7DT_C361K_HIGH_1x1')
    for key in telescope_keys:
        print(key)
        
        config = Configuration(telkey=key)
        config.register_telescope()
        print(config.config)
#%

# %%
