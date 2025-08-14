#%%
from ezphot.helper import Helper
from ezphot.utils import DataBrowser
from astropy.io import fits


REQUIRED_FIELDS = ['TELESCOP', 'INSTRUME', 'IMGTYPE', 'FILTER', 'OBJNAME']

class ImageFormatter(DataBrowser):
    def __init__(self, foldertype: str = 'scidata'):
        super().__init__(foldertype)
        self.target_pathlist = None
        self.succeeded_pathlist = []
        self.failed_pathlist = []

    # Check wheather the header is consistent with the defined kwargs
    def check_header(self, modify_header: bool = False, **kwargs):
        if self.target_pathlist is None:
            raise RuntimeError("No target pathlist found. Please run search_files() first.")
        self.succeeded_pathlist = []
        self.failed_pathlist = []
        for path in self.target_pathlist:
            try:
                header = fits.getheader(path)
                for key, value in kwargs.items():
                    if value is None:
                        if key not in header.keys():
                            print(f"Image {path} is missing required field: {key}")
                            if modify_header:
                                header[key] = value
                            self.failed_pathlist.append(path)
                            continue
                        else:
                            self.succeeded_pathlist.append(path)
                            continue
                    else:
                        if key not in header.keys():
                            print(f"Image {path} is missing required field: {key}")
                            if modify_header:
                                header[key] = value
                            self.failed_pathlist.append(path)
                            continue
                        else:
                            val_in_header = header[key]
                            if val_in_header != value:
                                print(f"Image {path} is not consistent with the defined kwargs: {key} = (Requested: {value}, Found: {val_in_header})")
                                if modify_header:
                                    header[key] = value
                                self.failed_pathlist.append(path)
                            else:
                                self.succeeded_pathlist.append(path)
                if modify_header:
                    for path in self.failed_pathlist:
                        data = fits.getdata(path)
                        fits.writeto(path, data=data, header=header, overwrite=True)
            except Exception as e:
                print(f"Error getting header for {path}: {e}")
                return False

    def search_files(self, pattern='*.fits', folder = None):
        # key = filter, value = list
        self.target_pathdict = self.search(pattern=pattern, return_type='path', folder = folder)
        # pathdict to list of Path objects
        self.target_pathlist = [p for paths in self.target_pathdict.values() for p in paths]

    
#%%
if __name__ == '__main__':
    formatter = ImageFormatter('refdata')
    formatter.observatory = 'undefined'
    formatter.search_files(pattern='*CBNUO*.fits', folder = '/home/hhchoi1022/data/refdata/undefined')
    formatter.check_header(modify_header=True, TELESCOP='CBNUO', IMGTYPE = 'LIGHT')
    
#%%