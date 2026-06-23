#%%
from bridge.connector import GWPortalConnector
#%%
gwconnector = GWPortalConnector('processed')
# %%

# tbl_recent = gwconnector.query(since_days = 550)
# tbl_recent = gwconnector.query(obs_start_date = '2025-01-01', obs_end_date = '2026-06-20')
#%%
tbl_recent = tbl_recent[tbl_recent['exptime'] == 100]
#%%
tbl_2026 = tbl_recent[[obstime.startswith('2026') for obstime in tbl_recent['obstime']]]
tbl_2025 = tbl_recent[[obstime.startswith('2025') for obstime in tbl_recent['obstime']]]
#%%

from astropy.stats import sigma_clip
import numpy as np
import pandas as pd

def sigma_clip_by_filter(df, column, sigma=3):

    clipped_rows = []

    for f in df['filter'].unique():

        sub = df[df['filter'] == f]

        data = sub[column].astype(float)

        mask = ~sigma_clip(data, sigma=sigma).mask

        clipped_rows.append(sub[mask])

    return pd.concat(clipped_rows)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = tbl_recent.to_pandas()
df_ul5 = sigma_clip_by_filter(df, 'ul5', sigma=3)

filter_order = [
'g','r','i','z',
'm400','m425','m450','m475','m500','m525','m550','m575',
'm600','m625','m650','m675','m700','m725','m750','m775',
'm800','m825','m850','m875'
]

sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

base_colors = sns.color_palette("Spectral", len(filter_order))

palette = dict(zip(filter_order, base_colors))

palette.update({
    'g': 'green',
    'r': 'red',
    'i': 'yellow',
    'z': 'black'
})

ax = sns.violinplot(
    data=df_ul5,
    x='filter',
    y='ul5',
    order=filter_order,
    palette=palette,
    inner=None,
    cut= 0,
    linewidth=1,
)

medians = df.groupby('filter')['ul5'].median()

for i, f in enumerate(filter_order):

    vals = df[df['filter']==f]['ul5']
    if len(vals)==0:
        continue

    median = medians[f]
    # ymax is 99 percentile of the values
    ymax = vals.quantile(0.999)

    ax.scatter(
        i, median,
        color='white',
        edgecolor='black',
        s=80,
        zorder=3
    )

    ax.text(
        i,
        median + 1.7,
        f"{median:.2f}",
        ha='center',
        fontsize=12,
        rotation=90,
        color='k'
    )

ax.set_ylim(15, 24)

ax.set_ylabel("Limiting magnitude (5σ)", fontsize = 14)
ax.set_xlabel("")
ax.set_title("Limiting magnitude distribution (exptime=100s)", fontsize = 16)

plt.xticks(rotation=45, fontsize = 12)
plt.yticks(fontsize = 12)

plt.tight_layout()
plt.show()
# %%


df = tbl_2026.to_pandas()
df_ul5 = sigma_clip_by_filter(df, 'seeing', sigma=3)

filter_order = [
'g','r','i','z',
'm400','m425','m450','m475','m500','m525','m550','m575',
'm600','m625','m650','m675','m700','m725','m750','m775',
'm800','m825','m850','m875'
]

sns.set_style("whitegrid")

plt.figure(figsize=(10,5))

base_colors = sns.color_palette("Spectral", len(filter_order))

palette = dict(zip(filter_order, base_colors))

palette.update({
    'g': 'green',
    'r': 'red',
    'i': 'yellow',
    'z': 'black'
})

ax = sns.violinplot(
    data=df_ul5,
    x='filter',
    y='seeing',
    order=filter_order,
    palette=palette,
    inner=None,
    cut=0,
    linewidth=1
)

medians = df_ul5.groupby('filter')['seeing'].median()

for i, f in enumerate(filter_order):

    vals = df_ul5[df_ul5['filter']==f]['seeing']
    if len(vals)==0:
        continue

    median = medians[f]
    # ymax is 99 percentile of the values
    ymax = vals.quantile(0.995)

    ax.scatter(
        i, median,
        color='white',
        edgecolor='black',
        s=80,
        zorder=3
    )

    ax.text(
        i,
        ymax + 0.6,
        f"{median:.2f}",
        ha='center',
        fontsize=12,
        rotation=90,
        color='k'
    )

ax.set_ylim(1.0, 7)
ax.set_ylabel("Seeing [arcsec]", fontsize = 14)
ax.set_xlabel("")
ax.set_title("Seeing distribution", fontsize = 16)

plt.xticks(rotation=45, fontsize = 12)
plt.yticks(fontsize = 12)

plt.tight_layout()
plt.show()
# %%
