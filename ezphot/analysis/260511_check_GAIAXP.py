#!/home/hhchoi1022/anaconda3/bin/python3.13

#%%
import pandas as pd
import numpy as np
import pyarrow.csv as pv
from pathlib import Path
import time

# ==================== Configuration ====================
BASE_DIR = Path('/lyman/data1/factory_doabdi/GAIA_BEST')
TILES_SUMMARY_PATH = BASE_DIR / 'summary.csv'
OUTPUT_DIR = Path('./tiles')

RA_COL = 'gaiaxp_photinfo_replenish_ra'
DEC_COL = 'gaiaxp_photinfo_replenish_dec'

EXCLUDE_COLUMNS = {
}

BLOCK_SIZE = 500 * 1024 * 1024  # 500 MB → ~37K rows per batch

CSV_FILES = sorted(BASE_DIR.glob('ra*.csv'))
print(f"Found {len(CSV_FILES)} CSV files: {[f.name for f in CSV_FILES]}")

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

print(f"  Polar / full-RA tiles : {tile_is_full_ra.sum()}")
print(f"  Degenerate-dec tiles  : {is_degenerate.sum()}")

# --- Dec-band spatial index (2° bands) ---
DEC_BAND_SIZE = 2.0
dec_bands: dict[int, np.ndarray] = {}
for i in range(n_tiles):
    lo = int(np.floor(tile_dec_min[i] / DEC_BAND_SIZE))
    hi = int(np.floor(tile_dec_max[i] / DEC_BAND_SIZE))
    for b in range(lo, hi + 1):
        dec_bands.setdefault(b, []).append(i)
for b in dec_bands:
    dec_bands[b] = np.array(dec_bands[b], dtype=np.int32)
print(f"  Dec bands             : {len(dec_bands)}")

# ==================== Core assignment function ====================
#%%
def assign_chunk_to_tiles(ra: np.ndarray, dec: np.ndarray):
    """Vectorised bounding-box assignment.  Returns {tile_index: row_indices}."""
    src_bands = np.floor(dec / DEC_BAND_SIZE).astype(np.int32)
    unique_bands = np.unique(src_bands)
    assignments: dict[int, np.ndarray] = {}

    for bk in unique_bands:
        if bk not in dec_bands:
            continue
        cand = dec_bands[bk]
        src_idx = np.where(src_bands == bk)[0]
        if len(src_idx) == 0:
            continue

        s_ra  = ra[src_idx]
        s_dec = dec[src_idx]

        # Vectorised dec check  (n_src, n_cand)
        dec_ok = ((s_dec[:, None] >= tile_dec_min[cand][None, :]) &
                  (s_dec[:, None] <= tile_dec_max[cand][None, :]))

        # Vectorised RA check  (n_src, n_cand)
        shifted = (s_ra[:, None] + tile_ra_shift[cand][None, :]) % 360.0
        ra_ok = (tile_is_full_ra[cand][None, :] |
                 ((shifted >= tile_ra_min_shifted[cand][None, :]) &
                  (shifted <= tile_ra_max_shifted[cand][None, :])))

        inside = dec_ok & ra_ok

        for j in range(len(cand)):
            mask_j = inside[:, j]
            if not mask_j.any():
                continue
            t_idx = cand[j]
            matched = src_idx[mask_j]
            if t_idx in assignments:
                assignments[t_idx] = np.concatenate([assignments[t_idx], matched])
            else:
                assignments[t_idx] = matched

    return assignments

# ==================== File processing ====================
#%%
def get_usable_columns(csv_path: Path) -> list[str]:
    """Read header only and return columns minus EXCLUDE_COLUMNS."""
    reader = pv.open_csv(csv_path, read_options=pv.ReadOptions(block_size=1024 * 1024))
    batch = next(reader)
    cols = [c for c in batch.schema.names if c not in EXCLUDE_COLUMNS]
    del batch, reader
    return cols


def process_file(csv_path: Path, tiles_with_headers: set):
    size_gb = csv_path.stat().st_size / 1e9
    print(f"\n{'=' * 60}")
    print(f"Processing  {csv_path.name}  ({size_gb:.1f} GB)")
    print(f"{'=' * 60}")

    usecols = get_usable_columns(csv_path)

    reader = pv.open_csv(
        csv_path,
        read_options=pv.ReadOptions(block_size=BLOCK_SIZE),
        convert_options=pv.ConvertOptions(include_columns=usecols),
    )

    total_rows = 0
    total_assignments = 0

    for chunk_num, batch in enumerate(reader):
        t0 = time.time()
        chunk = batch.to_pandas()
        n_rows = len(chunk)
        total_rows += n_rows

        ra  = chunk[RA_COL].values.astype(np.float64)
        dec = chunk[DEC_COL].values.astype(np.float64)

        valid = np.isfinite(ra) & np.isfinite(dec)
        assignments = assign_chunk_to_tiles(ra, dec)

        chunk_assign = 0
        chunk_tiles  = 0
        for t_idx, row_idx in assignments.items():
            row_idx = row_idx[valid[row_idx]]
            if len(row_idx) == 0:
                continue
            tid = tile_ids[t_idx]
            out = OUTPUT_DIR / f'{tid}.csv'
            write_hdr = tid not in tiles_with_headers
            chunk.iloc[row_idx].to_csv(out, mode='a', header=write_hdr, index=False)
            tiles_with_headers.add(tid)
            chunk_assign += len(row_idx)
            chunk_tiles  += 1

        total_assignments += chunk_assign
        dt = time.time() - t0
        print(f"  batch {chunk_num:4d} | {n_rows:>7,} rows | "
              f"{chunk_assign:>8,} assign → {chunk_tiles:>5,} tiles | {dt:.1f}s")

    del reader
    print(f"  ── subtotal: {total_rows:,} rows, {total_assignments:,} assignments")
    return total_rows, total_assignments

# ==================== Main ====================
#%%
if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles_with_headers: set = set()

    t_start = time.time()
    grand_rows = 0
    grand_assign = 0

    for csv_path in CSV_FILES:
        r, a = process_file(csv_path, tiles_with_headers)
        grand_rows  += r
        grand_assign += a

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE  {elapsed:.0f}s  ({elapsed / 60:.1f} min)")
    print(f"  rows processed   : {grand_rows:,}")
    print(f"  total assignments: {grand_assign:,}")
    print(f"  tiles with data  : {len(tiles_with_headers)}")
    print(f"  output dir       : {OUTPUT_DIR}")
    print(f"{'=' * 60}")
# %%



path = '~/code/ezphot/ezphot/analysis/tiles/T25468.csv'
# %%
tbl = Table.read(path, format='ascii')
# %%
len(tbl)
# %%
len(set(tbl['gaiaxp_photinfo_replenish_source_id']))
# %%
tbl.colnames

# %%
tbl['gaiaxp_photinfo_replenish_ra']
# %%
import matplotlib.pyplot as plt
plt.scatter(tbl['gaiaxp_photinfo_replenish_ra'], tbl['gaiaxp_photinfo_replenish_dec'])
plt.axvline(353.448162)
plt.axvline(355.642747)
plt.axhline(18.839864)
plt.axhline(20.212351)
plt.show()
# %%

# for i in range(len(tiles_df)):
i = 499
ra_list = [tiles_df['ra1'][i], tiles_df['ra2'][i], tiles_df['ra3'][i], tiles_df['ra4'][i]]
ra_min = min(ra_list)
ra_max = max(ra_list)
dec_list = [tiles_df['dec1'][i], tiles_df['dec2'][i], tiles_df['dec3'][i], tiles_df['dec4'][i]]
dec_min = min(dec_list)
dec_max = max(dec_list)
tile_id = tiles_df['id'][i]
path = f'~/code/ezphot/ezphot/analysis/tiles/{tile_id}.csv'
tbl = Table.read(path, format='ascii')
plt.figure(figsize=(10, 10))
plt.title(tile_id)
plt.scatter(tbl['gaiaxp_photinfo_replenish_ra'], tbl['gaiaxp_photinfo_replenish_dec'])
plt.axvline(ra_min)
plt.axvline(ra_max)
plt.axhline(dec_min)
plt.axhline(dec_max)
plt.show()
# %%
print('RA range: ', ra_min, ra_max)
print('Dec range: ', dec_min, dec_max)
print('Number of sources: ', len(tbl))
# %%