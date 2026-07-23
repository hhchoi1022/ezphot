# ezphot/__init__.py
__version__ = "0.4.16"

def initialize(configpath=None):
    """Initialize ezphot configuration.
    
    Creates the directory structure at ~/ezphot/ and copies default
    configuration files for all supported telescopes. This is called
    automatically on first use, but can also be called explicitly.
    
    Parameters
    ----------
    configpath : str or Path, optional
        Configuration directory. Defaults to ~/ezphot/config.
        
    Returns
    -------
    Configuration
        The initialized Configuration instance.
    """
    from pathlib import Path
    from .configuration import Configuration
    kwargs = {}
    if configpath is not None:
        kwargs['configpath'] = configpath
    config = Configuration(**kwargs)
    return config
