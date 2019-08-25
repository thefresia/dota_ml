#%%
from sklearn.manifold import TSNE
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from itertools import repeat
import numpy as np
import pymongo
import mglearn
from pprint import pprint
props = {"match_id", "barracks_status_dire", "barracks_status_radiant",
    "dire_score", "radiant_score", "duration", "first_blood_time", 
    "game_mode", "human_players", "radiant_win", "tower_status_dire",
    "tower_status_radiant","patch", "players"}
playerProps = { "assists", "backpack_0", "backpack_1", "backpack_2", "deaths",
    "denies","gold","gold_per_min","gold_spent","hero_damage","hero_healing",
    "hero_id","item_0","item_1","item_2","item_3","item_4","item_5",
    "kills", "last_hits","tower_damage","xp_per_min",
    "win","isRadiant",#"total_gold",
    "kda","benchmarks"}
learningProps = {"barracks_status_dire", "barracks_status_radiant",
    "dire_score", "radiant_score", "duration", "first_blood_time", 
    "game_mode", "radiant_win", "tower_status_dire",
    "tower_status_radiant"}
learningPlayerProps = { "assists", "deaths",
    "denies","gold","gold_per_min","gold_spent","hero_damage","hero_healing",
    "hero_id", "kills", "last_hits","tower_damage","xp_per_min",
    "win","isRadiant","kda"}
playerOnly = { "assists", "deaths",
    "denies","gold_per_min","hero_damage","hero_healing",
    "kills", "last_hits","tower_damage","xp_per_min",
    "win"

}

def transform(obj, properties):
    newObj = {}
    if obj == {}:
        return dict.fromkeys(properties, 0)
    for prop in properties:
        data = obj[prop]
        if (prop == "players"):
            for i in range(len(data)):
                data[i] = {} if 'leaver_status' in data[i] and (data[i]['leaver_status'] != 0 or data[i]['abandons'] !=0) else transform(data[i], playerProps) 
        newObj[prop] = data
    return newObj

class MongoDbConnection:
    @staticmethod
    def getCollection(db):
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient[db]
        return mydb

def train_hero():
    con = MongoDbConnection.getCollection("dota_ml")
    col = con['data']
    tsne = TSNE(random_state=69, n_components=2)
    count = 2000*10
    if (count > col.count()*10):
        exit
    data = np.empty((count,11))
    original={'heroNames': [None]*130, 'heroes': [None]*count,'tsne_transformed': [np.zeros(3) for i in repeat(None, 130)]}
    for hero in con["heroes"].find():
        original["heroNames"][hero['id']] = hero["localized_name"]
    i = 0
    for x in con["data"].find():
        #print(i)
        for player in x["players"]:
            if player == {}:
                continue
            heroId = player["hero_id"]
            data[i] = np.fromiter(transform(player, playerOnly).values(), dtype=float)
            original["heroes"][i] = {'name' :original["heroNames"][heroId], 'id':heroId}
            i+=1
            if i == count:
                break
        if i == count:
            break
    print('data is prepared.')
    print('applying tsne.')
    original["data"] = data
    #original["tsne"] = tsne.fit_transform(original["data"])
    print('tsne is finished.')
    res = original["data"]#original["tsne"]
    for i in range(len(res)):
        entry = original["tsne_transformed"][original["heroes"][i]['id']]
        entry[0] += res[i, 0]
        entry[1] += res[i, 1]
        entry[2] += 1
    plt.figure(figsize=(10, 10))
    res = original["tsne_transformed"]
    for i in range(len(res)):
        entry = res[i]
        if entry[2] != 0:
            entry[0]/=entry[2]
            entry[1]/=entry[2]
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    for entry in res:
        if (entry[0] > xmax):
            xmax = entry[0]
        if (entry[0] < xmin):
            xmin = entry[0]
        if (entry[1] > ymax):
            ymax = entry[1]
        if (entry[1] < ymin):
            ymin = entry[1]
    plt.xlim(xmin, xmax + 1)
    plt.ylim(ymin, ymax + 1)
    agg = AgglomerativeClustering(n_clusters=11, linkage='average')
    clusters = agg.fit_predict(res)
    colors = [ "red", "green", "blue", "purple", "white", "pink", "yellow", "orange", "brown", "grey", "x", "x" , "x" , "x" , "x"  ]
    group = []
    for c in colors:
        group.append([])
    for i in range(len(res)):
        if res[i][2] != 0:
            #plt.text(res[i][0], res[i][1], original["heroNames"][i], color=colors[clusters[i]])
            group[clusters[i]].append(original["heroNames"][i])
    plt.xlabel("t-SNE feature 0")
    plt.xlabel("t-SNE feature 1")
    for i in range(len(group)):
        print('%d.' % i, end = ' ')
        for hero in group[i]:
            print(hero, end = ', ')
        print('\n')
    print('tis over')

train_hero()