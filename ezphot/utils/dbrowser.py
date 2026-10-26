

#%%
import os
import re
import glob
import json
from pathlib import Path
from typing import List, Union

import requests
from astropy.table import Table

from ezphot.helper import Helper

#%%
class DBrowser:
    """
    DBrowser queries the GWPortal frame database (REST API) instead of globbing folders.
    It is the database counterpart of ``DataBrowser('scidata')``.

    Set the search attributes, then call ``search()``:

    >>> db = DBrowser('processed')
    >>> db.objname = 'T05785'; db.filter = 'm475'
    >>> tbl = db.search()                         # astropy Table of DB records
    >>> imgset = db.search(return_type='science') # ImageSet of the local scidata copies

    Credentials are read from ``~/ezphot/config/gwportal.config``
    (JSON: ``{"GWPORTAL_BASE_URL": "...", "GWPORTAL_API_KEY": "..."}``);
    environment variables of the same name take precedence.

    Notes
    -----
    - The API silently ignores parameters it does not know (or does not support for an endpoint)
      and then returns the whole table, so a search with no attribute set is refused.
    - ``objname`` is sent as ``tile_name`` (exact) for tile names ``T#####`` and as ``target_name``
      (substring on the server, made exact but case-insensitive here) otherwise; the ``tile``/``target``
      endpoints always get ``tile_name``/``target_name``.
    - ``ra``/``dec`` must be set together; a partial cone is ignored by the server.
    - ``night`` is the local night label; ``obs_start_date``/``obs_end_date`` are UTC dates of ``obstime``.
    - ``processed`` holds one row per processing version; ``version='latest'`` keeps the newest
      per raw frame, ``'v2'`` etc. keeps that version only, ``None`` keeps all.
    """

    QUERY_TYPES = {'raw': 'frames/raw/',
                   'processed': 'frames/processed/',
                   'combined': 'frames/combined/',
                   'tile': 'tiles/',
                   'target': 'targets/'}
    PAGE_SIZE = 5000

    def __init__(self, query_type: str = 'processed'):
        if query_type not in self.QUERY_TYPES:
            raise ValueError(f"Invalid query_type: {query_type}. Choose from {list(self.QUERY_TYPES)}.")
        self.helper = Helper()
        self.query_type = query_type
        self.basepath = Path(self.helper.config['SCIDATA_DIR'])
        self.base_url, self.api_key = self._load_credentials()
        self._session = None
        # Search attributes
        self.objname = None         # tile or target name, e.g. 'T05785', 'EG21'
        self.telname = None         # unit, e.g. '7DT02' or '7DT02,7DT03'
        self.filter = None          # e.g. 'm475' or 'm450,m475'
        self.night = None           # 'YYYY-MM-DD' local night label
        self.obs_start_date = None  # 'YYYY-MM-DD' (UTC obstime)
        self.obs_end_date = None
        self.since_days = None
        self.ra = None              # cone search centre and radius [deg]
        self.dec = None
        self.radius = None
        self.version = 'latest'

    def __repr__(self):
        txt = f"<DBrowser(query_type='{self.query_type}', base_url='{self.base_url}')>\n"
        txt += "Search Attributes:\n"
        for key in ['objname', 'telname', 'filter', 'night', 'obs_start_date', 'obs_end_date', 'since_days', 'ra', 'dec', 'radius', 'version']:
            txt += f"  {key:<14} : {getattr(self, key) if getattr(self, key) is not None else '*'}\n"
        return txt

    def search(self, return_type: str = 'table') -> Union[Table, List[str]]:
        """
        Query the database with the current attributes.

        Parameters
        ----------
        return_type : str
            ``'table'``, ``'path'`` or ``'science'``.

        Returns
        -------
        output : Table, list, ImageSet
            - ``'table'``: ``astropy.table.Table`` of DB records, with a ``local_path`` column
              (``''`` when the frame is not in the local scidata) for raw/processed frames.
            - ``'path'``: list of local scidata paths that exist.
            - ``'science'``: ``ImageSet`` of ``ScienceImage`` loaded from those paths.
        """
        if return_type not in ('table', 'path', 'science'):
            raise ValueError(f"Invalid return_type: {return_type}. Choose from 'table', 'path', 'science'.")
        if return_type != 'table' and self.query_type not in ('raw', 'processed'):
            raise ValueError(f"return_type='{return_type}' needs query_type 'raw' or 'processed'.")
        # '' is treated as unset: the server ignores empty params and returns the whole table
        objname, ra, dec, radius = [None if v == '' else v for v in (self.objname, self.ra, self.dec, self.radius)]
        if (ra is None) != (dec is None) or (radius is not None and ra is None):
            raise ValueError("Set ra and dec together (radius optional: without it, a point-in-footprint search).")
        if self.query_type == 'tile':
            is_tile = True
        elif self.query_type == 'target':
            is_tile = False
        else:
            is_tile = objname is not None and re.fullmatch(r'T\d{5}', objname, flags=re.IGNORECASE) is not None
        frame_params = {'unit_name': self.telname,
                        'filter_name': self.filter,
                        'night_date': self.night,
                        'date_start': self.obs_start_date,
                        'date_end': self.obs_end_date,
                        'days': self.since_days}
        frame_params = {k: v for k, v in frame_params.items() if v not in (None, '')}
        if self.query_type in ('tile', 'target') and frame_params:
            raise ValueError(f"{list(frame_params)} only apply to frame queries (raw, processed, combined).")
        params = {'tile_name': objname.upper() if is_tile and objname is not None else None,
                  'target_name': objname if not is_tile else None,
                  'ra': ra,
                  'dec': dec,
                  'radius': radius}
        params = dict({k: v for k, v in params.items() if v is not None}, **frame_params)
        if not params:
            raise ValueError("Set at least one search attribute; an unfiltered query returns the whole database.")

        rows = self._fetch_all(self.QUERY_TYPES[self.query_type], params)
        if objname is not None and not is_tile:
            # target_name is a case-insensitive substring match on the server
            namekey = 'name' if self.query_type == 'target' else 'target'
            rows = [row for row in rows if str(row[namekey]).lower() == objname.lower()]
        if self.query_type == 'processed' and self.version is not None:
            rows = self._select_version(rows, self.version)
        print(f"[INFO] Found {len(rows)} {self.query_type} records for {params}")

        tbl = Table(rows=rows) if rows else Table()
        if self.query_type not in ('raw', 'processed'):
            return tbl

        if len(tbl):
            tbl['local_path'] = [self._local_path(row) for row in rows]
        if return_type == 'table':
            return tbl

        paths = [p for p in tbl['local_path'] if p] if len(tbl) else []
        if len(paths) < len(tbl):
            print(f"[WARNING] {len(tbl) - len(paths)} of {len(tbl)} frames are not in the local scidata ({self.basepath})")
        if return_type == 'path':
            return paths

        from ezphot.imageobjects import ImageSet
        from ezphot.utils import DataBrowser
        return ImageSet(DataBrowser('scidata')._to_science_images(paths))

    def _load_credentials(self):
        configfile = self.helper.path_config / 'gwportal.config'
        config = json.loads(configfile.read_text()) if configfile.exists() else dict()
        base_url = os.getenv('GWPORTAL_BASE_URL') or config.get('GWPORTAL_BASE_URL')
        api_key = os.getenv('GWPORTAL_API_KEY') or config.get('GWPORTAL_API_KEY')
        if not base_url or not api_key:
            raise ValueError(f"GWPortal credentials not found. Set GWPORTAL_BASE_URL/GWPORTAL_API_KEY in {configfile} or in the environment.")
        return base_url.rstrip('/') + '/', api_key

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({'X-API-Key': self.api_key, 'Accept': 'application/json'})
        return self._session

    def _fetch_all(self, endpoint: str, params: dict) -> List[dict]:
        url = self.base_url + 'api/' + endpoint
        rows, page, num_pages = [], 1, 1
        while page <= num_pages:
            response = self.session.get(url, params=dict(params, page=page, page_size=self.PAGE_SIZE), timeout=60)
            response.raise_for_status()
            result = response.json()
            rows.extend(result.get('results', []))
            num_pages = result.get('num_pages', 1)
            page += 1
        return rows

    @staticmethod
    def _select_version(rows: List[dict], version: str) -> List[dict]:
        if version != 'latest':
            return [row for row in rows if row['processing_version'] == version]
        def _key(v):
            return tuple(int(x) for x in str(v).lstrip('v').split('.'))
        latest = dict()
        for row in rows:
            frame = row['raw_frame_id']
            if frame not in latest or _key(row['processing_version']) > _key(latest[frame]['processing_version']):
                latest[frame] = row
        return list(latest.values())

    def _local_path(self, row: dict) -> str:
        objname = row.get('object_name') or row.get('tile') or row.get('target')
        matched = glob.glob(str(self.basepath / '7DT' / '*' / str(objname) / row['unit'] / row['filter'] / row['unified_filename']))
        return matched[0] if matched else ''
