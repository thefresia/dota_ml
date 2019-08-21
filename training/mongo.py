import pymongo
import train
from train import MongoDbConnection

con = MongoDbConnection.getCollection("dota_ml")
col = con['heroes']
i = 1
with open('C://Users//i2k//Desktop//dota_ml//data//heroes.txt') as f:
    content = f.readlines()
    for h in content:
        col.insert_one({'_id' : i, 'hero':h.strip()})
        i+=1