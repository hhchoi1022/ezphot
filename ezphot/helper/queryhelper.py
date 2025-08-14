#%%
import time
import requests
import os
import re
import io
from astropy.table import Table
import pandas as pd

#%%

class Queryhelper():
    def query_ztf_lightcurve(self, ra, dec, radius, tname = None, return_output=False):
        
        base_url = f'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE {ra} {dec} {radius}&BAD_CATFLAGS_MASK=32768'
        # Sending the GET request
        response = requests.get(base_url)
        # Check if the request was successful
        if response.status_code == 200:
            # Process the response content
            with open(".tmpdata", "w") as f:
                f.write(response.text)
            from astropy.io.votable import parse
            votable = parse(".tmpdata").get_first_table()
            tbl = votable.to_table()
            os.remove(".tmpdata")
            if tname is None:
                tname = "%.3f_%.3f_%.3f".format(tbl.meta['RA'], tbl.meta['DEC'], tbl.meta['RADIUS'])
            folder = f"data/ZTF/{tname}"
            fname = os.path.join(folder, f"{tname}.csv") 
            os.makedirs(folder, exist_ok=True)
            tbl.write(fname, format="ascii.csv", overwrite=True)
            if return_output:
                print('Table: ', tbl)
                return tbl
            else:
                return fname
        else:
            print(f"Error: {response.status_code}")
            return
        
    def _request_atlas_token(self, username = 'hhchoi1022', password = 'lksdf1020!'):

        
        BASEURL = "https://fallingstar-data.com/forcedphot"
        resp = requests.post(url=f"{BASEURL}/api-token-auth/", data={'username': f"{username}", 'password': f"{password}"})

        if resp.status_code == 200:
            token = resp.json()['token']
            print(f'Your token is {token}')
            headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
        else:
            print(f'ERROR {resp.status_code}')
            print(resp.json())
    
    def query_atlas_lightcurve(self, ra, dec, radius, 
                               tname = None, 
                               mjd_min : float = None,
                               mjd_max : float = None,
                               return_output = False, 
                               token = '810625b718b73b9ebac070be6c61a353d6d9eaa7'):

        BASEURL = "https://fallingstar-data.com/forcedphot"
        headers = {'Authorization': f'Token {token}', 'Accept': 'application/json'}
        task_url = None
        while not task_url:
            with requests.Session() as s:
                url = f"{BASEURL}/queue/"
                data = {'ra': ra, 'dec': dec}
                if mjd_min:
                    data['mjd_min'] = mjd_min
                if mjd_max:
                    data['mjd_max'] = mjd_max

                resp = s.post(url, headers=headers, data=data)

                if resp.status_code == 201:  # successfully queued
                    task_url = resp.json()['url']
                    print(f'The task URL is {task_url}')
                elif resp.status_code == 429:  # throttled
                    message = resp.json()["detail"]
                    print(f'{resp.status_code} {message}')
                    t_sec = re.findall(r'available in (\d+) seconds', message)
                    t_min = re.findall(r'available in (\d+) minutes', message)
                    if t_sec:
                        waittime = int(t_sec[0])
                    elif t_min:
                        waittime = int(t_min[0]) * 60
                    else:
                        waittime = 10
                    print(f'Waiting {waittime} seconds')
                    time.sleep(waittime)
                else:
                    print(f'ERROR {resp.status_code}')
                    print(resp.json())
                    sys.exit() 
        result_url = None
        while not result_url:
            with requests.Session() as s:
                resp = s.get(task_url, headers=headers)

                if resp.status_code == 200:  # HTTP OK
                    if resp.json()['finishtimestamp']:
                        result_url = resp.json()['result_url']
                        print(f"Task is complete with results available at {result_url}")
                        break
                    elif resp.json()['starttimestamp']:
                        print(f"Task is running (started at {resp.json()['starttimestamp']})")
                    else:
                        print("Waiting for job to start. Checking again in 10 seconds...")
                    time.sleep(10)
                else:
                    print(f'ERROR {resp.status_code}')
                    print(resp.json())
                    sys.exit()
        with requests.Session() as s:
            textdata = s.get(result_url, headers=headers).text
        dfresult = pd.read_csv(io.StringIO(textdata.replace("###", "")), delim_whitespace=True)
        tbl = Table.from_pandas(dfresult) 
        if tname is None:
            tname = f"{ra}_{dec}_{radius}"
        folder = f"data/ATLAS/{tname}"
        fname = os.path.join(folder, f"{tname}.csv") 
        os.makedirs(folder, exist_ok=True)
        tbl.write(fname, format="ascii.csv", overwrite=True)    
        
        if return_output:
            print('Table: ', tbl)
            return tbl
        else:
            return fname

# %% Example
# self = Queryhelper()
# tbl =self.query_atlas_lightcurve(41.575542, -30.239489, 0.0028, mjd_min=60180.0, mjd_max=60300.0, return_output=True)
