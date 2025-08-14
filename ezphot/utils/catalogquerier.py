

#%%
import inspect
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.imcce import Skybot
#%%


class CatalogQuerier:
    """
    Class to handle external data queries using Astroquery Vizier.
    This class is a wrapper for the Astroquery Vizier and Skybot classes.
    This class providies 
    
    1. Vizier catalog query (SDSS, GAIA, 2MASS, AllWISE, PS1)
        - SDSS: SDSS DR17
        - GAIA: Gaia DR3
        - 2MASS: 2MASS All-Sky Point Source Catalog
        - AllWISE: AllWISE Data Release
        - PS1: PanSTARRS DR1
    2. Skybot catalog query (SKYBOT)
        - SKYBOT: Skybot
    
    """
    
    def __init__(self, catalog_key: str = None):
        self.skybot = Skybot
        self.vizier = Vizier()
        self.vizier.ROW_LIMIT = 1000
        self.vizier.columns = ['*', "+_r"]
        self.vizier.TIMEOUT = 60
        self.vizier.column_filters = {}
        if (catalog_key not in self.catalog_ids.keys()) and catalog_key is not None:
            raise ValueError(f"Catalog Key '{catalog_key}' is not recognized. Available keys: {list(self.catalog_ids.keys())}")
        self.current_catalog_key = catalog_key

    def __repr__(self):
        return f"CatalogQuerier(catalog={self.current_catalog_key})\n{self.config}\n For help, use 'help(self)' or `self.help()`."

    def help(self):
        # Get all public methods from the class, excluding `help`
        methods = [
            (name, obj)
            for name, obj in inspect.getmembers(self.__class__, inspect.isfunction)
            if not name.startswith("_") and name != "help"
        ]

        # Build plain text list with parameters
        lines = []
        for name, func in methods:
            sig = inspect.signature(func)
            params = [str(p) for p in sig.parameters.values() if p.name != "self"]
            sig_str = f"({', '.join(params)})" if params else "()"
            lines.append(f"- {name}{sig_str}")

        # Final plain text output
        help_text = ""
        print(f"Help for {self.__class__.__name__}\n{help_text}\n\nPublic methods:\n" + "\n".join(lines))

    def query(self,
              coord,
              epoch = None,
              radius_arcsec = 3600):
        """
        Query a specific catalog around given coordinates.
        
        This method will search for the catalog in the catalog defined by the current_catalog_key.
        If the current_catalog_key is 'SKYBOT', it will use the Skybot catalog query.
        Otherwise, it will use the Vizier catalog query.
        
        Parameters
        ----------
        coord : SkyCoord
            The coordinates to query.
        epoch : Time, optional
            The epoch to query.
        radius_arcsec : float, optional
            The radius to query in arcseconds.
            
        Returns
        -------
        Table
            The catalog table.
        """
        
        
        if self.current_catalog_key is 'SKYBOT':
            return self._query_skybot_catalog(coord, epoch, radius_arcsec=radius_arcsec)
        else:
            return self._query_vizier_catalog(coord, radius_arcsec)
    
    def show_available_catalogs(self):
        """
        Display available catalogs.
        
        Returns
        -------
        list of catalog keys: list
            List of available catalogs.
        """
        print("Current catalog: ", self.current_catalog_key)
        print("Available catalogs\n==================")
        for catalog_name, catalog_id in self.catalog_ids.items():
            print(f"{catalog_name}: {catalog_id}")
        return list(self.catalog_ids.keys())
    
    def change_catalog(self, catalog_key):
        """
        Change the current catalog to query.
        
        Parameters
        ----------
        catalog_key : str
            The catalog key to change to.
        
        Returns
        -------
        None
        """
        if catalog_key in self.catalog_ids.keys():
            self.current_catalog_key = catalog_key
            print(self.__repr__())
        else:
            raise ValueError(f"Catalog Key '{catalog_key}' is not recognized.")

    def _query_vizier_catalog(self,
                             coord, 
                             radius_arcsec=10):
        """Query a specific catalog around given coordinates.
        
        Parameters
        ----------
        coord : SkyCoord
            The coordinates to query.
        radius_arcsec : float, optional
            The radius to query in arcseconds.
        """
        if type(radius_arcsec) is not u.Quantity:
            radius_arcsec = radius_arcsec * u.arcsec
        print(f'Starting query for catalog {self.current_catalog_key} around {coord} with radius {radius_arcsec}')
        print(f'{self.config}')
        result = self.vizier.query_region(coord, radius=radius_arcsec, catalog=self.current_catalog_id)
        print(f'Query completed. Found {len(result)} records.')
        return result
    
    def _query_skybot_catalog(self,
                             coord,
                             epoch = None,
                             radius_arcsec = 3600):
        """Query a specific region using Skybot.
        
        Parameters
        ----------
        coord : SkyCoord
            The coordinates to query.
        """ 
        if type(radius_arcsec) is not u.Quantity:
            radius_arcsec = radius_arcsec * u.arcsec
        if epoch is None:
            epoch = Time.now()
        print(f'Starting Skybot query around {coord} with radius {radius_arcsec}')
        result = self.skybot.cone_search(coord, radius_arcsec, epoch, location = 500)
        return result

    @property
    def config(self):
        """
        Get the configuration of the CatalogQuerier.
        
        Returns
        -------
        Configuration
            The configuration of the CatalogQuerier.
        """
        class Configuration:
            """Handles configuration for Vizier queries."""

            def __init__(self, vizier_instance: Vizier):
                self._vizier = vizier_instance

            @property
            def row_limit(self):
                return self._vizier.ROW_LIMIT

            @row_limit.setter
            def row_limit(self, value):
                self._vizier.ROW_LIMIT = value

            @property
            def columns(self):
                return self._vizier.columns

            @columns.setter
            def columns(self, value):
                self._vizier.columns = value

            @property
            def timeout(self):
                return self._vizier.TIMEOUT

            @timeout.setter
            def timeout(self, value):
                self._vizier.TIMEOUT = value

            @property
            def filters(self):
                return self._vizier.column_filters

            @filters.setter
            def filters(self, filters: dict):
                self._vizier.column_filters = filters

            def reset(self):
                self._vizier.ROW_LIMIT = -1
                self._vizier.columns = ['*']

            def __repr__(self):
                return (f"========Vizier Configuration========\n"
                        f"  row_limit       = {self.row_limit}\n"
                        f"  columns         = {self.columns}\n"
                        f"  filters         = {self.filters}\n"
                        f"  timeout         = {self.timeout} s"
                        f"\n====================================")
        
        return Configuration(self.vizier)                

    @property
    def current_catalog_id(self):
        """
        Get the current catalog ID.
        
        Returns
        -------
        str
            The current catalog ID.
        """
        if self.current_catalog_key is None:
            return None
        else:
            return self.catalog_ids[self.current_catalog_key]
    
    @property
    def catalog_ids(self):
        """
        Get the available catalog IDs.
        
        Returns
        -------
        dict
            The catalog IDs.
        """
        catalog_ids = dict()
        catalog_ids['GAIA'] = 'I/355'
        catalog_ids['GAIA_DR3'] = "I/355/gaiadr3"
        catalog_ids['GAIA_DR3_SPEC'] = 'I/355/spectra'
        catalog_ids['GAIAXP'] = 'I/355/xpsample'      
        # 2MASS (Final release)
        catalog_ids['2MASS'] = "II/246/out"

        # AllWISE (All-sky WISE data)
        catalog_ids['AllWISE'] = "II/328/allwise"

        # Pan-STARRS DR1 (Stacked photometry)
        catalog_ids['PS1'] = "II/349/ps1"

        # SDSS DR17 (Photometric data)
        catalog_ids['SDSS'] = "V/167/sdss17"
        
        # Skybot
        catalog_ids['SKYBOT'] = "IMCCE/Skybot"
        return catalog_ids
        
#%%
if __name__ == "__main__":
    from astropy.time import Time
    # Example usage
    self = CatalogQuerier(catalog_key='SKYBOT')
    
    # Query a specific region
    coord = SkyCoord(ra=233.322342*u.deg, dec=-68.007909*u.deg, frame='icrs')
    epoch = Time('2025-02-09T00:00:00')
    #result = ed.query_catalog(coord, radius_arcsec=5)
    
    result = self.query(coord = coord, epoch = epoch)
# %%
