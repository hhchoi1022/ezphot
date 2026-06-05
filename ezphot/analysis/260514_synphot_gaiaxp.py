#%%
import pandas as pd
import numpy as np
import pyarrow.csv as pv
from pathlib import Path

# ==================== Configuration ====================
GAIAXP_BEST_yk_DIR = Path('/lyman/data1/factory_doabdi/GAIA_BEST')
GAIAXP_BEST_hh_DIR = Path('/lyman/data1/factory_doabdi/GAIA_tiles')
TILES_SUMMARY_PATH = GAIAXP_BEST_yk_DIR / 'summary.csv'
GAIAXP_BEST_yk_DIR = GAIAXP_BEST_yk_DIR / 'tile'
RA_COL = 'gaiaxp_photinfo_replenish_ra'
DEC_COL = 'gaiaxp_photinfo_replenish_dec'
GAIAXP_original_DIR = Path('/lyman/data1/factory/ref_cat')
# ==================== Load & precompute tile geometry ====================
#%%
print("Loading tile summary...")
tiles_df = pd.read_csv(TILES_SUMMARY_PATH)
tiles_df = tiles_df.loc[:, ~tiles_df.columns.str.startswith('Unnamed')]

corner_cols = ['ra1', 'dec1', 'ra2', 'dec2', 'ra3', 'dec3', 'ra4', 'dec4']
for col in corner_cols:
    tiles_df[col] = pd.to_numeric(tiles_df[col], errors='coerce')
tiles_df = tiles_df.dropna(subset=corner_cols).reset_index(drop=True)
n_tiles = len(tiles_df)
print(f"  {n_tiles} tiles loaded")

tile_ids = tiles_df['id'].values
tile_ra_center = tiles_df['ra'].values.astype(np.float64)
tile_dec_center = tiles_df['dec'].values.astype(np.float64)

tile_ra_corners = np.column_stack([
    tiles_df['ra1'].values, tiles_df['ra2'].values,
    tiles_df['ra3'].values, tiles_df['ra4'].values,
]).astype(np.float64)
tile_dec_corners = np.column_stack([
    tiles_df['dec1'].values, tiles_df['dec2'].values,
    tiles_df['dec3'].values, tiles_df['dec4'].values,
]).astype(np.float64)

# --- Dec bounds ---
tile_dec_min_corner = tile_dec_corners.min(axis=1)
tile_dec_max_corner = tile_dec_corners.max(axis=1)

tile_dec_min = tile_dec_min_corner.copy()
tile_dec_max = tile_dec_max_corner.copy()

is_degenerate = (tile_dec_max_corner - tile_dec_min_corner) < 0.01
for i in np.where(is_degenerate)[0]:
    reflected = 2.0 * tile_dec_center[i] - tile_dec_min_corner[i]
    tile_dec_min[i] = max(min(tile_dec_min_corner[i], reflected), -90.0)
    tile_dec_max[i] = min(max(tile_dec_max_corner[i], reflected),  90.0)

# --- RA bounds (shifted so tile center → 180°) ---
tile_ra_shift = 180.0 - tile_ra_center
shifted_corners = (tile_ra_corners + tile_ra_shift[:, None]) % 360.0
tile_ra_min_shifted = shifted_corners.min(axis=1)
tile_ra_max_shifted = shifted_corners.max(axis=1)
tile_is_full_ra = (tile_ra_max_shifted - tile_ra_min_shifted) > 180.0
#%%
from astropy.table import Table
tile_name = 'T00017'
BEST_path = GAIAXP_BEST_yk_DIR / f'{tile_name}.csv'
BEST_hh_path = GAIAXP_BEST_hh_DIR / f'{tile_name}.csv'
original_path = GAIAXP_original_DIR / f'gaiaxp_dr3_synphot_{tile_name}.csv'
tbl_GAIA_yk = Table.read(BEST_path, format='ascii')
tbl_GAIA_hh = Table.read(BEST_hh_path, format='ascii')
tbl_original = Table.read(original_path, format='ascii')
#%%
print(len(tbl_GAIA_yk))
print(len(tbl_GAIA_hh))
print(len(tbl_original))
#%%
colnames = ['gaiaxp_photinfo_replenish_source_id',
 'gaiaxp_photinfo_replenish_ra',
 'gaiaxp_photinfo_replenish_dec',
 'gaiaxp_photinfo_replenish_parallax',
 'gaiaxp_photinfo_replenish_pmra',
 'gaiaxp_photinfo_replenish_pmdec',
 'gaiaxp_photinfo_replenish_bp_mag',
 'gaiaxp_photinfo_replenish_rp_mag',
 'gaiaxp_photinfo_replenish_g_mag',
 'gaiaxp_photinfo_replenish_ruwe',
 'gaiaxp_photinfo_replenish_excess_factor',
 'correctedxpspecv1_flux',
 'correctedxpspecv1_flux_error',
 'correctedxpspecv1_flux_cor',
 'correctedxpspecv1_c2',
 'correctedxpspecv1_c3',
 'correctedxpspecv1_caution']

tbl_GAIA_yk.rename_columns(tbl_GAIA_yk.colnames, colnames)
tbl_GAIA_hh.rename_columns(tbl_GAIA_hh.colnames, colnames)

ra_yk = []
dec_yk = []
ra_hh = []
dec_hh = []
ra_original = []
dec_original = []
for i,row in enumerate(tbl_GAIA_yk):
    try:
        ra = float(row['gaiaxp_photinfo_replenish_ra'])
        dec = float(row['gaiaxp_photinfo_replenish_dec'])
        ra_yk.append(ra)
        dec_yk.append(dec)
    except:
        print(i, row['gaiaxp_photinfo_replenish_ra'], row['gaiaxp_photinfo_replenish_dec'])
for i,row in enumerate(tbl_GAIA_hh):
    try:
        ra = float(row['gaiaxp_photinfo_replenish_ra'])
        dec = float(row['gaiaxp_photinfo_replenish_dec'])
        ra_hh.append(ra)
        dec_hh.append(dec)
    except:
        print(i, row['gaiaxp_photinfo_replenish_ra'], row['gaiaxp_photinfo_replenish_dec'])
for i,row in enumerate(tbl_original):
    try:
        ra = float(row['ra'])
        dec = float(row['dec'])
        ra_original.append(ra)
        dec_original.append(dec)
    except:
        print(i, row['ra'], row['dec'])
#%%
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 10))
plt.scatter(ra_yk, dec_yk, c = 'k', alpha = 0.1)
plt.scatter(ra_hh, dec_hh, c = 'r', alpha = 0.1)
# plt.scatter(ra_original, dec_original, c = 'b', alpha = 0.1)

# %%
row = tiles_df.iloc[17]
ra_all = [row['ra'], row['ra1'], row['ra2'], row['ra3'], row['ra4']]
dec_all = [row['dec'], row['dec1'], row['dec2'], row['dec3'], row['dec4']]
ra_min = min(ra_all)
ra_max = max(ra_all)
dec_min = min(dec_all)
dec_max = max(dec_all)
# %%
plt.figure(figsize = (10, 10))
plt.scatter(ra_hh, dec_hh, c = 'r', alpha = 0.1, label = 'GAIAXP (HH)')
plt.scatter(ra_yk, dec_yk, c = 'k', alpha = 0.1, label = 'GAIAXP (YK)')
plt.axvline(ra_min)
plt.axvline(ra_max)
plt.axhline(dec_min)
plt.axhline(dec_max)
plt.scatter(row['ra1'], row['dec1'], c = 'b', marker = '*', s = 50)
plt.scatter(row['ra2'], row['dec2'], c = 'b', marker = '*', s = 50)
plt.scatter(row['ra3'], row['dec3'], c = 'b', marker = '*', s = 50)
plt.scatter(row['ra4'], row['dec4'], c = 'b', marker = '*', s = 50)
print(row['ra1'], row['ra2'], row['ra3'], row['ra4'])
print(row['dec1'], row['dec2'], row['dec3'], row['dec4'])
plt.legend()
plt.show()
# %%
print(ra_max - ra_min, dec_max - dec_min)
# %%
