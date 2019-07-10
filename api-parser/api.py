import requests
import os
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup

opendotaURL = 'https://api.opendota.com/api/'
headers = {
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'
    }
folderSize = 100

def handler429():
    time.sleep(10)

def createSession():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def folderName(i):
    return i / folderSize * folderSize

def proceed(response, i, directory):
    r = response
    if (r.status_code == 429):
        handler429()
    bad = r.status_code == 404 or r.status_code == 429
    data = str(r.status_code) + ' ' + str(i)
    if bad:
        print('BAD'+ data)
    else:
        print('OK' + data)
        with open(directory+str(i)+'.json', "w") as f:
            f.write(r.text.encode('ascii', 'ignore').decode('ascii'))
    time.sleep(0.3)
    return int(bad)

def opendota(amount, start = 4887481309):
    if (amount < 0):
        print('invalid param amount')
        return
    url = opendotaURL + 'matches/'
    missing = 0
    current = start
    session = createSession()
    last = start - amount
    while (current > last):
        folder = folderName(current)
        directory = os.path.dirname(os.getcwd())+'/data/matches/'+str(folder)+'/'    
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            current = folder-1
            continue
        while (current >= folder):
            matchUrl = url + str(current)
            r = session.get(matchUrl, headers = headers)
            missing+=proceed(r, current, directory)
            current -= 1
        print(str((100  - (current-last)/float(amount) * 100)) + '%')
    print(str(missing)+'/'+str(amount))

opendota(1000)