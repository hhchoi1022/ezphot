#%%
from tippy.helper import PhotometryHelper
from tippy.helper import AnalysisHelper
from tippy.helper import OperationHelper
import inspect
#%%
class Helper(PhotometryHelper, AnalysisHelper):
    def __init__(self):
        self.operation = OperationHelper()
        super().__init__()
        self.__doc__ = self._generate_doc()
    
    def __repr__(self):
        cls = self.__class__
        methods = [
            f'{cls.__name__}.{name}()\n'
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not name.startswith('_') and method.__qualname__.startswith(cls.__name__)
        ]
        return '[Methods]\n' + ''.join(methods)
    
    def _generate_doc(self):
        import inspect
        lines = [f"{self.__class__.__name__} methods:\n", "=" * 40]
        for name, method in self.__class__.__dict__.items():
            if callable(method) and not name.startswith("_"):
                sig = inspect.signature(method)
                lines.append(f"- {name}{sig}")
        return "\n".join(lines)

