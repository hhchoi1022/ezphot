


#%%

import requests
from astropy.io.votable import parse
import io
import pandas as pd

# Define the target coordinates and search radius
ra = 189.99763  # Right Ascension in degrees (example: Sombrero Galaxy)
dec = -11.62305  # Declination in degrees
radius = 0.17  # Size in degrees
band = "g,r,i"  # Filter bands

# Construct the query URL (METADATA returns an image list)
query_url = f"https://api.skymapper.nci.org.au/public/siap/dr4/query?POS={ra},{dec}&SIZE={radius}&BAND={band}&FORMAT=image/fits"

# Request metadata
response = requests.get(query_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse VOTable XML response
    votable = parse(io.BytesIO(response.content))
    table = votable.get_first_table().to_table()
    print("Query successful! Check the table above for available images.")

    return table 
    
    
else:
    print(f"Error: Unable to fetch image list. Status code: {response.status_code}")
    
#%%