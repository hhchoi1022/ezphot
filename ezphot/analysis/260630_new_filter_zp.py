
#%%
from bridge.connector import GWPortalConnector
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from astropy.stats import sigma_clip

#%%
gwconnector = GWPortalConnector('processed')
tbl = gwconnector.query(since_days=60)
df = tbl.to_pandas()
df['obstime'] = pd.to_datetime(df['obstime'])
df['seeing'] = df['seeing'].astype(float)
df = df[df['exptime'] == 100]

#%% Sigma-clip seeing per filter to remove outliers
clipped = []
for f in df['filter'].unique():
    sub = df[df['filter'] == f]
    mask = ~sigma_clip(sub['seeing'], sigma=3).mask
    clipped.append(sub[mask])
df_clean = pd.concat(clipped)

#%% Plot 1: Seeing vs time (all data, colored by filter)
fig, ax = plt.subplots(figsize=(14, 5))
for filt, grp in df_clean.groupby('filter'):
    ax.scatter(grp['obstime'], grp['seeing'], s=8, alpha=0.4, label=filt)
ax.set_ylabel('Seeing [arcsec]', fontsize=13)
ax.set_xlabel('Date', fontsize=13)
ax.set_title('Seeing over Time (exptime=100s, 3σ-clipped)', fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, ncol=2)
plt.tight_layout()
plt.show()

#%% Plot 2: Nightly median seeing trend
df_clean['night'] = df_clean['obstime'].dt.date
nightly = df_clean.groupby('night')['seeing'].agg(['median', 'std', 'count']).reset_index()
nightly['night'] = pd.to_datetime(nightly['night'])

fig, ax = plt.subplots(figsize=(14, 5))
ax.errorbar(nightly['night'], nightly['median'], yerr=nightly['std'],
            fmt='o-', ms=5, capsize=2, color='steelblue', ecolor='lightblue', alpha=0.8)
ax.axhline(nightly['median'].median(), color='red', ls='--', alpha=0.6,
           label=f"Overall median = {nightly['median'].median():.2f}\"")
ax.set_ylabel('Median Seeing [arcsec]', fontsize=13)
ax.set_xlabel('Night', fontsize=13)
ax.set_title('Nightly Median Seeing', fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()

#%% Best nights: rank by median seeing (lower is better)
best_nights = nightly[nightly['count'] >= 10].sort_values('median').head(20)
print("=== Top 20 Best Seeing Nights (>=10 exposures) ===")
print(best_nights[['night', 'median', 'std', 'count']].to_string(index=False))

#%% Details for the single best night
best_date = best_nights.iloc[0]['night']
best_df = df_clean[df_clean['obstime'].dt.date == best_date.date()]
print(f"\n=== Best Night: {best_date.date()} ===")
print(f"  Median seeing : {best_df['seeing'].median():.2f}\"")
print(f"  Mean seeing   : {best_df['seeing'].mean():.2f}\"")
print(f"  Min seeing    : {best_df['seeing'].min():.2f}\"")
print(f"  N exposures   : {len(best_df)}")
print(f"  Filters used  : {sorted(best_df['filter'].unique())}")
#%% From the best night, select 10 tile_ids randomly. 
# random_tile_ids = best_df['tile'].sample(n=10)
random_tile_ids = [
    'T21033',
    'T19067',
    'T19920',
    'T20208',
    'T19067',
    'T20145',
    'T20147',
    'T19310',
    'T19318',
    'T20747'
]
tile_id = random_tile_ids[0]
#%%
from bridge.alertmonitor import AlertProcessor
from bridge.objects import Alert
from astropy.time import Time
for tile_id in random_tile_ids:
    tile_df = best_df[best_df['tile'] == tile_id]
    print(tile_id, len(tile_df))
    ra_center = tile_df['ra_center'].mean()
    dec_center = tile_df['dec_center'].mean()
    alert_instance = Alert(objname = tile_id, trigger_time = Time('2026-05-01'))
    alertprocessor = AlertProcessor()
    alertprocessor.load_images_db(alert_instance = alert_instance)
    alertprocessor.config.photcal['catalog_type'] = 'GAIAXP'
    alertprocessor.config.photcal['catalog_version'] = 'v1'
    alertprocessor.config.photcal['radius_arcsec'] = None
    alertprocessor.config.single_process['do_preprocess'] = True
    alertprocessor.config.single_process['do_platesolve'] = True
    alertprocessor.config.single_process['do_calculate_bkgrms_from_propagation'] = False
    alertprocessor.config.single_process['do_stack'] = False
    alertprocessor.pipeline_before_stacking(alert_instance)
#%%
from bridge.configuration import Configuration
config = Configuration('alertprocessor.config')
mag_range_dict = config.photcal['mag_range_dict']

from ezphot.imageobjects import ScienceImage
failed_dict = dict()
for path in tile_df['filepath']:
    try:
        target_img = ScienceImage(path=path)
        target_img.write()
        target_srcmask = target_img.calculate_sourcemask(save = True, visualize = False)
        target_bkg = target_img.calculate_bkg(target_srcmask=target_srcmask, save=True, visualize=False)
        target_bkgrms = target_img.calculate_bkgrms(target_srcmask=target_srcmask, save=True, visualize=False)
        target_catalog = target_img.photometry_sex(target_bkg=target_bkg, target_bkgrms=target_bkgrms, save=True, visualize=False, detection_sigma = 1.5)

        target_img.photometric_calibration(
            target_catalog = target_catalog,
            catalog_type = 'GAIAXP_CORR_LAMOST',
            catalog_version = 'v2',
            mag_lower = mag_range_dict[target_img.filter][0],
            mag_upper = mag_range_dict[target_img.filter][1],
            visualize = False,
            save_fig = True,
        )
    except Exception as e:
        failed_dict[path] = e
# %%
from ezphot.utils import CatalogQuerier
catalog_querier = CatalogQuerier('GAIAXP')
# %%
from ezphot.utils import DataBrowser
dbrowser = DataBrowser('scidata')
dbrowser.objname = tile_id
# %%
target_imgset = dbrowser.search('*.fits', return_type='science')
target_catalogset = dbrowser.search('*.cat', return_type='catalog')
# %%
from ezphot.dataobjects import PhotometricSpectrum
# %%
ra = 248.6337387
dec = 5.9413066
#%% Raodomly select 10 targets with magnitude < 18
%matplotlib inline
from astropy.coordinates import SkyCoord
from ezphot.dataobjects import Spectrum
target_tbl = target_catalogset.catalogs[0].data
target_tbl_bright = target_tbl[(target_tbl['MAGSKY_APER_3'] < 18) & (target_tbl['MAGSKY_APER_3'] > 11) & (target_tbl['X_IMAGE'] < 9000) & (target_tbl['X_IMAGE'] > 1000) & (target_tbl['Y_IMAGE'] < 6000) & (target_tbl['Y_IMAGE'] > 1000)]
random_idx = np.random.choice(len(target_tbl_bright), size=50, replace=False)
target_tbl_bright_random = target_tbl_bright[random_idx]
all_target_rows = []
for target_row in target_tbl_bright_random:
    ra = target_row['X_WORLD']
    dec = target_row['Y_WORLD']
    try:
        catalog_querier.config.timeout = 10
        result = catalog_querier.query(coord = SkyCoord(ra=ra, dec=dec, unit='deg'), radius_arcsec=5)
        if len(result) > 0:
            photspec = PhotometricSpectrum(catalogset=target_catalogset)
            fig, _, ax, data_tbl = photspec.plot(ra=ra, dec=dec, matching_radius_arcsec=10, flux_key = 'MAGSKY_APER_3', fluxerr_key = 'MAGERR_APER_3')
            fig = list(fig.values())[0]
            ax = list(ax.values())[0]
            spec = Spectrum(wavelength = np.array(result[0]['lambda']), flux = np.array(result[0]['Flux']), fluxerr = np.array(result[0]['e_Flux']), flux_unit= 'flamb_si')
            synphot_result = spec.synphot(filterset= 'medium')[0]
            ax.plot(spec.wavelength.value/10, spec.ab.flux.value)
            for filter_, val in synphot_result.items():
                if filter_ in data_tbl['filter']:
                    ax.scatter(val['wl_pivot'].value, val['mag'], facecolor = 'r', edgecolor='k', alpha=0.5, zorder=1, s = 100)
                        
            # ------------------------------------------------------------
            # Update synthetic magnitude, synthetic mag error, and mag diff
            # mag_diff = observed_mag - synthetic_mag
            # ------------------------------------------------------------

            # Choose the filter column name in data_tbl
            if "filter" in data_tbl.colnames:
                filter_col = "filter"
            elif "FILTER" in data_tbl.colnames:
                filter_col = "FILTER"
            else:
                raise KeyError(f"No filter column found. Available columns: {data_tbl.colnames}")

            # Add columns if they do not exist yet
            for colname in ["syn_mag", "syn_magerr", "mag_diff"]:
                if colname not in data_tbl.colnames:
                    data_tbl[colname] = np.full(len(data_tbl), np.nan)

            # Update data_tbl
            for filter_, syn in synphot_result.items():

                # synthetic magnitude
                syn_mag = float(np.asarray(syn["mag"]))

                # synthetic magnitude error
                # depending on synphot_result structure, the key may differ
                if "magerr" in syn:
                    syn_magerr = float(np.asarray(syn["magerr"]))
                elif "mag_err" in syn:
                    syn_magerr = float(np.asarray(syn["mag_err"]))
                elif "e_mag" in syn:
                    syn_magerr = float(np.asarray(syn["e_mag"]))
                else:
                    syn_magerr = np.nan

                # find matching filter row in observed photometry table
                idx = np.where(np.asarray(data_tbl[filter_col]).astype(str) == str(filter_))[0]

                if len(idx) == 0:
                    continue

                # observed magnitude
                obs_mag = np.asarray(data_tbl["MAGSKY_APER_3"][idx]).astype(float)

                # update table
                data_tbl["syn_mag"][idx] = syn_mag
                data_tbl["syn_magerr"][idx] = syn_magerr
                data_tbl["mag_diff"][idx] = obs_mag - syn_mag
            # Calculate chisq
            data_tbl['total_error'] = np.sqrt(data_tbl['MAGERR_APER_3']**2 + data_tbl['syn_magerr']**2 + data_tbl['ZPERR_APER_3']**2)
            data_tbl['chisq'] = (data_tbl['mag_diff'] / data_tbl['total_error'])**2

            # ------------------------------------------------------------
            # Combine chi-square values by filter
            # ------------------------------------------------------------

            if "filter" in data_tbl.colnames:
                filter_col = "filter"
            elif "FILTER" in data_tbl.colnames:
                filter_col = "FILTER"
            else:
                raise KeyError(f"No filter column found. Available columns: {data_tbl.colnames}")

            rows = []

            for filt in np.unique(data_tbl[filter_col]):

                sub = data_tbl[data_tbl[filter_col] == filt]

                # Use only valid rows
                valid = np.isfinite(sub["chisq"])
                sub = sub[valid]

                if len(sub) == 0:
                    continue

                chisq_sum = np.sum(sub["chisq"])
                n_data = len(sub)

                # If no fitted parameter per filter, dof = n_data
                chisq_red = chisq_sum / n_data

                rows.append(
                    dict(
                        filter=filt,
                        n_data=n_data,
                        chisq_sum=chisq_sum,
                        chisq_red=chisq_red,
                    )
                )
            from astropy.table import Table

            chisq_filter_tbl = Table(rows=rows)

            # ------------------------------------------------------------
            # Add wl_pivot to chisq_filter_tbl
            # ------------------------------------------------------------

            if "wl_pivot" not in chisq_filter_tbl.colnames:
                chisq_filter_tbl["wl_pivot"] = np.full(len(chisq_filter_tbl), np.nan)

            for i, row in enumerate(chisq_filter_tbl):
                filt = row["filter"]

                if filt in synphot_result:
                    chisq_filter_tbl["wl_pivot"][i] = synphot_result[filt]["wl_pivot"].value


            # ------------------------------------------------------------
            # Create top chi-square panel above the main SED axis
            # ------------------------------------------------------------

            pos = ax.get_position()

            # Shrink the main SED panel downward
            ax.set_position([
                pos.x0,
                pos.y0,
                pos.width,
                pos.height * 0.72,
            ])

            # Create top chi-square axis
            ax_chisq = fig.add_axes([
                pos.x0,
                pos.y0 + pos.height * 0.76,
                pos.width,
                pos.height * 0.22,
            ], sharex=ax)

            # Hide x tick labels on top panel
            ax_chisq.tick_params(labelbottom=False)
            # ------------------------------------------------------------
            # Plot reduced chi-square per filter
            # ------------------------------------------------------------

            ax_chisq.scatter(
                chisq_filter_tbl["wl_pivot"],
                chisq_filter_tbl["chisq_red"],
                s=100,
                facecolor="none",
                edgecolor="k",
                zorder=5,
            )

            ax_chisq.axhline(1.0, ls="--", color="gray", alpha=0.6)

            ax_chisq.set_ylabel(r"$\chi^2_\nu$")
            ax_chisq.set_title(
                rf"Coordinate(RA, Dec) = ({ra:.4f}, {dec:.4f}), Total mean $\chi^2$ = {np.nanmean(chisq_filter_tbl['chisq_red']):.2f}",
                fontsize=13,
            )
            ylim = ax_chisq.get_ylim()
            yrange = ylim[1] - ylim[0]

            for row in chisq_filter_tbl:
                ax_chisq.text(
                    row["wl_pivot"],
                    row["chisq_red"] + yrange * 0.1,
                    f'{row["filter"]}: {row["chisq_red"]:.1f}',
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
            ax_chisq.set_ylim(ylim[0], ylim[1] + yrange * 0.2)


            all_target_rows.append(target_row)
            fig
    except Exception as e:
        print(e)
#%%
fig
#%% Plot Zeropoint error for all catalogs

#%%
all_catalogs = []
from ezphot.utils import DataBrowser
for tile_id in random_tile_ids:
    dbrowser = DataBrowser('scidata')
    dbrowser.objname = tile_id
    # target_imgset = dbrowser.search('*.fits', return_type='science')
    target_catalogset = dbrowser.search('*.cat', return_type='catalog')
    catalogs = target_catalogset.catalogs
    all_catalogs.extend(catalogs)
#%%
from ezphot.dataobjects import CatalogSet
target_catalogset = CatalogSet(all_catalogs)
#%% Plot Zeropoint error for all catalogs
import numpy as np
import matplotlib.pyplot as plt
from ezphot.dataobjects import PhotometricSpectrum
wl_dict = PhotometricSpectrum.FILTER_PIVOT_WAVELENGTH_NM


#%%
all_zp_err = []
all_color_term = []
all_filter = []
all_pivot_wavelength = []

for catalog in target_catalogset.catalogs:
    try:
        zperr = catalog.data["ZPERR_APER_3"][0]
        filt = catalog.info.filter
        color_slope = catalog.target_img.header['K_COLOR_APER_3_g-r']
        all_color_term.append(color_slope)

        all_zp_err.append(zperr)
        all_filter.append(filt)
        all_pivot_wavelength.append(wl_dict[filt])
    except Exception as e:
        print(e)

all_zp_err = np.array(all_zp_err)
all_filter = np.array(all_filter)
all_pivot_wavelength = np.array(all_pivot_wavelength)
all_color_term = np.array(all_color_term)

# %%
#%% Original scatter + sigma-clipped median ± std for zeropoint error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from ezphot.dataobjects import LightCurve
color_dict = LightCurve.FILTER_COLOR

circle_filters = [f"m{wl}" for wl in range(400, 876, 25)]

plot_tbl = pd.DataFrame({
    "filter": all_filter,
    "wl": all_pivot_wavelength,
    "zperr": all_zp_err,
})

fig, ax = plt.subplots(figsize=(10, 5))

sigma = 1.0

# --------------------------------------------------
# 1. Plot original scatter points
# --------------------------------------------------
for filt, wl, zperr in zip(all_filter, all_pivot_wavelength, all_zp_err):

    if filt in circle_filters:
        marker = "o"
    else:
        marker = "*"

    ax.scatter(
        wl,
        zperr,
        marker=marker,
        s=60,
        color=color_dict[filt],
        edgecolor="none",
        alpha=0.25,
        zorder=1,
    )

# --------------------------------------------------
# 2. Plot sigma-clipped median ± std per filter
# --------------------------------------------------
for filt, group in plot_tbl.groupby("filter"):

    wl = np.array(group["wl"], dtype=float)
    zperr = np.array(group["zperr"], dtype=float)

    valid = np.isfinite(wl) & np.isfinite(zperr)
    wl = wl[valid]
    zperr = zperr[valid]

    if len(zperr) == 0:
        continue

    clipped = sigma_clip(
        zperr,
        sigma=sigma,
        maxiters=5,
        cenfunc="median",
        stdfunc="std",
    )

    keep = ~clipped.mask

    wl_clip = wl[keep]
    zperr_clip = zperr[keep]

    if len(zperr_clip) == 0:
        continue

    wl_med = np.nanmedian(wl_clip)
    zperr_med = np.nanmedian(zperr_clip)
    zperr_std = np.nanstd(zperr_clip, ddof=1) if len(zperr_clip) > 1 else 0.0

    if filt in circle_filters:
        marker = "o"
        label = "Original filters"
    else:
        marker = "*"
        label = "New filters"

    ax.errorbar(
        wl_med,
        zperr_med,
        yerr=zperr_std,
        fmt=marker,
        markersize=13,
        color=color_dict[filt],
        markeredgecolor="k",
        markeredgewidth=1.0,
        ecolor=color_dict[filt],
        elinewidth=2,
        capsize=4,
        alpha=1.0,
        zorder=3,
        label=label,
    )

# --------------------------------------------------
# 3. Legend and labels
# --------------------------------------------------
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

ax.set_xlabel("Pivot wavelength [nm]", fontsize=14)
ax.set_ylabel("Zeropoint error", fontsize=14)
ax.set_ylim(0.0, 0.1)

ax.grid(alpha=0.3)

fig

#%% Original scatter + sigma-clipped median ± std
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from ezphot.dataobjects import LightCurve
color_dict = LightCurve.FILTER_COLOR

circle_filters = [f"m{wl}" for wl in range(400, 876, 25)]

plot_tbl = pd.DataFrame({
    "filter": all_filter,
    "wl": all_pivot_wavelength,
    "color_slope": all_color_term,
})

fig, ax = plt.subplots(figsize=(10, 5))

sigma = 2

# --------------------------------------------------
# 1. Plot original scatter points
# --------------------------------------------------
for filt, wl, color_slope in zip(all_filter, all_pivot_wavelength, all_color_term):

    if filt in circle_filters:
        marker = "o"
    else:
        marker = "*"

    ax.scatter(
        wl,
        color_slope,
        marker=marker,
        s=60,
        color=color_dict[filt],
        edgecolor="none",
        alpha=0.25,
        zorder=1,
    )

# --------------------------------------------------
# 2. Plot sigma-clipped median ± std per filter
# --------------------------------------------------
for filt, group in plot_tbl.groupby("filter"):

    wl = np.array(group["wl"], dtype=float)
    color_slope = np.array(group["color_slope"], dtype=float)

    valid = np.isfinite(wl) & np.isfinite(color_slope)
    wl = wl[valid]
    color_slope = color_slope[valid]

    if len(color_slope) == 0:
        continue

    clipped = sigma_clip(
        color_slope,
        sigma=sigma,
        maxiters=2,
        cenfunc="median",
        stdfunc="std",
    )

    keep = ~clipped.mask

    wl_clip = wl[keep]
    slope_clip = color_slope[keep]

    if len(slope_clip) == 0:
        continue

    wl_med = np.nanmedian(wl_clip)
    slope_med = np.nanmedian(slope_clip)
    slope_std = np.nanstd(slope_clip, ddof=1) if len(slope_clip) > 1 else 0.0

    if filt in circle_filters:
        marker = "o"
        label = "Original filters"
    else:
        marker = "*"
        label = "New filters"

    ax.errorbar(
        wl_med,
        slope_med,
        yerr=slope_std,
        fmt=marker,
        markersize=15,
        color=color_dict[filt],
        markeredgecolor="k",
        markeredgewidth=1.0,
        ecolor=color_dict[filt],
        elinewidth=2,
        capsize=4,
        alpha=1.0,
        zorder=3,
        label=label,
    )

# --------------------------------------------------
# 3. Legend and labels
# --------------------------------------------------
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

ax.axhline(0, color="k", ls="--", alpha=0.4)

ax.set_xlabel("Pivot wavelength [nm]", fontsize=14)
ax.set_ylabel("Color slope", fontsize=14)
ax.set_ylim(-0.2, 0.2)

ax.grid(alpha=0.3)

plt.show()
# %%
fig
# %%
