from db.connection import MongoDbConnection
import os

con = MongoDbConnection.getCollection("dota_ml")
col = con['heroes']
i = 1
with open(os.getcwd() + '/data/heroes.txt') as f:
    content = f.readlines()
    for h in content:
        col.insert_one({'_id' : i, 'hero':h.strip()})
        i+=1