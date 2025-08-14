#%%
from pathlib import Path
import json
from typing import Union
import shutil

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
        if self.telkey:
            self.path_config_specific_telescope = self.path_config_specific / self.telkey
            self._configfiles_telescopes = list(self.path_config_specific_telescope.glob('*.config'))
            if not self._configfiles_telescopes:
                print('No configuration file is found.\nTo make default configuration files, run tcspy.configuration.make_config')
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
            self.telkey = tel_key
            self.path_config_specific_telescope = self.path_config_specific / self.telkey
            self.path_config_specific_telescope.mkdir(parents=True, exist_ok=True)
            self._register_telescope()
        
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
            SEX_CONFIG = str(self.path_config_global / 'sextractor' / f'{self.telkey}.sexconfig'),
            SEX_CONFIGDIR = str(self.path_config_global / 'sextractor'),
            SEX_LOGDIR = str(self.path_log / 'sextractor' / 'log'),
            SEX_HISTORYDIR = str(self.path_log / 'sextractor' / 'history')
        )
        astrometry_config = dict(
            ASTROMETRY_SEXCONFIG = str(self.path_config_global / 'sextractor' / f'{self.telkey}.astrometry.sexconfig')
        )
        scamp_config = dict(
            SCAMP_CONFIG = str(self.path_config_global / 'scamp' / 'default.scampconfig'),
            SCAMP_SEXCONFIG = str(self.path_config_global / 'sextractor' / f'{self.telkey}.scamp.sexconfig'),
            SCAMP_CONFIGDIR = str(self.path_config_global / 'scamp'),
            SCAMP_LOGDIR = str(self.path_log / 'scamp' / 'log'),
            SCAMP_HISTORYDIR = str(self.path_log / 'scamp' / 'history')
        )
        swarp_config = dict(
            SWARP_CONFIG = str(self.path_config_global / 'swarp' / f'{self.telkey}.swarpconfig'),
            SWARP_CONFIGDIR = str(self.path_config_global / 'swarp'),
            SWARP_LOGDIR = str(self.path_log / 'swarp' / 'log'),
            SWARP_HISTORYDIR = str(self.path_log / 'swarp' / 'history')
        )
        psfex_config = dict(
            PSFEX_CONFIG = str(self.path_config_global / 'psfex' / 'default.psfexconfig'),
            PSFEX_SEXCONFIG = str(self.path_config_global / 'sextractor' / f'{self.telkey}.psfex.sexconfig'),
            PSFEX_CONFIGDIR = str(self.path_config_global / 'psfex'),
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
            OBSERVATORY_TELESCOPEINFO = str(self.path_config_global / 'CCD.dat')
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
        sex_config.pop('SEX_CONFIG', None)
        scamp_config.pop('SCAMP_SEXCONFIG', None)
        swarp_config.pop('SWARP_CONFIG', None)
        psfex_config.pop('PSFEX_SEXCONFIG', None)

        for cfg, name in [
            (sex_config, 'sex.config'),
            (scamp_config, 'scamp.config'),
            (swarp_config, 'swarp.config'),
            (psfex_config, 'psfex.config'),
        ]:
            self._make_configfile(cfg, name, self.path_config_global)
#%%
if __name__ == '__main__':
    # For the first time, initiailize the configuration
    config = Configuration()
    config.initialize(copy_default=True)
    
    telescope_keys = [
        '7DT_C361K_HIGH_1x1', '7DT_C361K_HIGH_2x2', '7DT_C361K_LOW_1x1', '7DT_C361K_LOW_2x2',
        'CBNUO_STX16803_1x1', 'LSGT_SNUCAMII_1x1', 'LSGT_ASI1600MM_1x1',
        'RASA36_KL4040_HIGH_1x1', 'RASA36_KL4040_MERGE_1x1', 'SAO_C361K_1x1',
        'SOAO_FLI4K_1x1', 'KCT_STX16803_1x1', 'SkyMapper_SG_32_Det_1x1']
    
    for key in telescope_keys:
        print(key)
        config = Configuration(telkey=key)
        config.register_telescope()
        #print(config.config)

# %%
