#%%
from tippy.helper import PhotometryHelper
from Research.helper import SpectroscopyHelper
from Research.helper import AnalysisHelper
import inspect
#%%
class Helper(PhotometryHelper, SpectroscopyHelper, AnalysisHelper):
    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        methods = [f'Helper.{name}()\n' for name, method in inspect.getmembers(
            Helper, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt
