import requests
import os
import time
import json
import pymongo
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pprint import pprint
from bs4 import BeautifulSoup

class OpendotaApi:
    opendotaURL = 'https://api.opendota.com/api/'
    headers = {
        'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'
        }
    folderSize = 100
    code404 = 0
    code429 = 0
    good = 0
    ts = 0

    def __init__(self):
        return

    def handler429(self):
        self.code429+=1
        time.sleep(10)

    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s %s/%s good: %s 404: %s 429: %s' % (prefix, bar, percent, suffix, iteration, total, self.good, self.code404, self.code429), end = '\r')
        if iteration == total: 
            print()

    @staticmethod
    def createSession():
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    @staticmethod
    def folderName(i):
        return i // OpendotaApi.folderSize * OpendotaApi.folderSize

    def proceed(self, response, i, directory):
        r = response
        if r.status_code == 429:
            self.handler429()
        if r.status_code == 404:
            self.code404+=1
        if r.status_code == 200:
            self.good+=1
            with open(directory+str(i)+'.json', "w") as f:
                f.write(r.text.encode('ascii', 'ignore').decode('ascii'))
        time.sleep(0.3)

    def saveLog(self, start, amount):
        directory = os.path.dirname(os.getcwd())+'/data/logs/' 
        with open(directory+str(datetime.now()).replace(' ','_').replace(':','-')+'.txt', "w") as f:
                f.write('Start match: ' + str(start))
                f.write('\nStart time: ' + str(self.ts))
                f.write('\nFinished at: ' + str(datetime.now()))
                f.write('\nTotal sec.: ' + str((self.ts-datetime.now()).total_seconds()))
                f.write('\nAmount: ' + str(amount))
                f.write('\nGood: ' + str(self.good))
                f.write('\n404: ' +str(self.code404))
                f.write('\n429: ' + str(self.code429))

    def opendota(self, amount, start = 4887481309):
        if (amount < 0):
            print('invalid param amount')
            return
        self.ts = datetime.now()
        url = self.opendotaURL + 'matches/'
        current = start
        session = self.createSession()
        last = start - amount
        while (current > last):
            self.printProgressBar(start - current, amount, prefix= 'Progress:', suffix='Proceeded:',length=50)
            folder = self.folderName(current)
            directory = os.path.dirname(os.getcwd())+'/data/matches/'+str(folder)+'/'    
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                current = folder-1
                continue
            folder = folder if folder > last else last
            for current in range(current, folder,-1):
                matchUrl = url + str(current)
                r = session.get(matchUrl, headers = self.headers)
                self.proceed(r, current, directory)
                current -= 1
                self.printProgressBar(start - current, amount, prefix= 'Progress:', suffix='Proceeded:',length=50)
        self.saveLog(start, amount)
    
    def remove_none(self, obj):
        if isinstance(obj, (list, tuple, set)):
            return type(obj)(self.remove_none(x) for x in obj if x is not None)
        elif isinstance(obj, dict):
            return type(obj)((self.remove_none(k), self.remove_none(v))
            for k, v in obj.items() if k is not None and v is not None)
        else:
            return obj

    def proceedMatches(self):
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["dota_ml"]
        col = db["data"]

        rawDir = os.path.dirname(os.getcwd())+'/dota_ml/data/matches/'
        for matchFolder in os.listdir(rawDir):
            folder = rawDir+matchFolder+'/'
            if len(os.listdir(folder)) == 0:
                print(folder +' is empty')
                os.rmdir(folder)
                continue
            for match in os.listdir(folder):
                matchPath = matchFolder + '/' + match 
                with open(rawDir+matchPath) as f:
                    data = json.load(f)
                    if not (col.find_one({'match_id': data['match_id']}) is None) or not data['game_mode'] in [22,2,3,4,5]:
                        continue
                    print('inserting %d'%data['match_id'])
                    col.insert_one(self.remove_none(data))

caller = OpendotaApi()
#caller.opendota(50000, 4895900000)
caller.proceedMatches()