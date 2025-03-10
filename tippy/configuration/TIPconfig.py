# Written by Hyeonho Choi 2023.01
# %%
import glob
import os
from astropy.io import ascii
import json

class TIPConfig:
    def __init__(self,
                 telkey: str = None,
                 configpath : str = f'{os.path.dirname(os.path.abspath(__file__))}',
                 **kwargs):
        self.telkey = telkey
        self.config = dict()
                
        # global config params
        self.path_global = configpath
        self.path_home = os.path.expanduser('~')
        self._configfilekey_global = os.path.join(self.path_global, '*.config')
        self._configfiles_global = glob.glob(self._configfilekey_global)
        config_global = self.load_configuration(self._configfiles_global)
        self.config.update(config_global)
        
        if self.telkey:
            # Specified units config params
            self.path_telescope = os.path.join(configpath, self.telkey)
            self._configfilekey_unit = os.path.join(self.path_telescope, '*.config')
            self._configfiles_unit = glob.glob(self._configfilekey_unit)
            if len(self._configfiles_unit) == 0:
                print('No configuration file is found.\nTo make default configuration files, run tcspy.configuration.make_config')
            else:
                config_unit = self.load_configuration(self._configfiles_unit)
                self.config.update(config_unit)

    def load_configuration(self, 
                            configfiles):
        all_config = dict()
        for configfile in configfiles:
            with open(configfile, 'r') as f:
                config = json.load(f)
                all_config.update(config)
        return all_config

    def make_configfile(self, 
                        dict_params : dict,
                        filename: str,
                        savepath : str):
        filepath = os.path.join(savepath, filename)
        with open(filepath, 'w') as f:
            json.dump(dict_params, f, indent=4)
        print('New configuration file made : %s' % (filepath))
        
    def initialize_config(self):
        savepath_tel = self.path_telescope
        savepath_global = self.path_global
        if not os.path.exists(savepath_tel):
            os.makedirs(savepath_tel, exist_ok=True)
            
        # LOCAL CONFIGURATION
        sex_config = dict(SEX_CONFIG = f'{self.path_global}/sextractor/{self.telkey}.sexconfig',
                          SEX_CONFIGDIR = f'{self.path_global}/sextractor/',
                          SEX_LOGDIR = f'{self.path_home}/code/sextractor/log/',
                          SEX_HISTORYDIR = f'{self.path_home}/code/sextractor/history/')
        scamp_config = dict(SCAMP_CONFIG = f'{self.path_global}/scamp/default.scampconfig',
                            SCAMP_SEXCONFIG = f'{self.path_global}/sextractor/{self.telkey}.scamp.sexconfig',
                            SCAMP_CONFIGDIR = f'{self.path_global}/scamp/',
                            SCAMP_LOGDIR = f'{self.path_home}/code/scamp/log/',
                            SCAMP_HISTORYDIR = f'{self.path_home}/code/scamp/history/')
        swarp_config = dict(SWARP_CONFIG = f'{self.path_global}/swarp/{self.telkey}.swarpconfig',
                            SWARP_CONFIGDIR = f'{self.path_global}/swarp/',
                            SWARP_LOGDIR = f'{self.path_home}/code/swarp/log/',
                            SWARP_HISTORYDIR = f'{self.path_home}/code/swarp/history/'
                            )
        psfex_config = dict(PSFEX_CONFIG = f'{self.path_global}/psfex/default.psfexconfig',
                            PSFEX_SEXCONFIG = f'{self.path_global}/sextractor/{self.telkey}.psfex.sexconfig',
                            PSFEX_CONFIGDIR = f'{self.path_global}/psfex/',
                            PSFEX_LOGDIR = f'{self.path_home}/code/psfex/log/',
                            PSFEX_HISTORYDIR = f'{self.path_home}/code/psfex/history/')
        
        self.make_configfile(sex_config, filename='sex.config', savepath = savepath_tel)
        self.make_configfile(scamp_config, filename='scamp.config', savepath = savepath_tel)
        self.make_configfile(swarp_config, filename='swarp.config', savepath = savepath_tel)
        self.make_configfile(psfex_config, filename='psfex.config', savepath = savepath_tel)
        
        # GLOBAL CONFIGURATION        
        calibdata_config = dict(CALIBDATA_DIR = f'{self.path_home}/data/calibdata',
                                CALIBDATA_MASTERDIR = f'{self.path_home}/data/mastercalib')
        refdata_config = dict(REFDATA_DIR = f'{self.path_home}/data/refdata')
        scidata_config = dict(SCIDATA_DIR = f'{self.path_home}/data/scidata')
        catalog_config = dict(CATALOG_DIR = f'{self.path_global}/../catalog/catalog_archive/')
        
        observatory_config = dict(OBSERVATORY_LOCATIONINFO = f'{self.path_global}/obs_location.txt',
                                  OBSERVATORY_TELESCOPEINFO = f'{self.path_global}/CCD.txt')
        
        sdtdata_config = dict(SDTDATA_OBSSOURCEDIR = f'/lyman/data1/obsdata/',
                              SDTDATA_OBSDESTDIR = f'/home/hhchoi1022/data/obsdata/7DT/',
                              SDTDATA_SCISOURCEDIR = f'/lyman/data1/processed_1x1_gain2750/',
                              SDTDATA_SCIDESTDIR = f'/home/hhchoi1022/data/scidata/7DT/7DT_C361K_HIGH_1x1')
                    

        # GLOBAL CONFIGURATION
        del sex_config['SEX_CONFIG']
        del scamp_config['SCAMP_SEXCONFIG']
        del swarp_config['SWARP_CONFIG']
        del psfex_config['PSFEX_SEXCONFIG']
        self.make_configfile(calibdata_config, filename='calibdata.config', savepath = savepath_global)
        self.make_configfile(refdata_config, filename='refdata.config', savepath = savepath_global)
        self.make_configfile(scidata_config, filename='scidata.config', savepath = savepath_global)
        self.make_configfile(catalog_config, filename='catalog.config', savepath = savepath_global)
        self.make_configfile(observatory_config, filename='observatory.config', savepath = savepath_global)
        self.make_configfile(sdtdata_config, filename='sdtdata.config', savepath = savepath_global)

        self.make_configfile(sex_config, filename='sex.config', savepath = savepath_global)
        self.make_configfile(scamp_config, filename='scamp.config', savepath = savepath_global)
        self.make_configfile(swarp_config, filename='swarp.config', savepath = savepath_global)
        self.make_configfile(psfex_config, filename='psfex.config', savepath = savepath_global)
#%%
if __name__ == '__main__':
    os.listdir('./')
    telescope_keys = ['7DT_C361K_HIGH_1x1',
                      '7DT_C361K_HIGH_2x2',
                      '7DT_C361K_LOW_1x1',
                      '7DT_C361K_LOW_2x2',
                      'CBNUO_STX16803_1x1',
                      'LSGT_SNUCAMII_1x1',
                      'LSGT_ASI1600MM_1x1',
                      'RASA36_KL4040_HIGH_1x1',
                      'RASA36_KL4040_MERGE_1x1',
                      'SAO_C361K_1x1',
                      'SOAO_FLI4K_1x1',
                      'KCT_STX16803_1x1']
    for key in telescope_keys:
        print(key)
        config = TIPConfig(telkey = key)
        config.initialize_config()
        print(config.config)
# %%
