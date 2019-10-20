#%%
from db.connection import MongoDbConnection

con = MongoDbConnection.getCollection("dota_ml")
col = con['data']

matches = col.count()
radiant = col.count({'radiant_win' : True})
print(matches)
print(radiant)
print(matches/radiant)

#%%
