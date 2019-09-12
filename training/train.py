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

def validateModel(amount):
    pass

def train_hero():
    con = MongoDbConnection.getCollection("dota_ml")
    col = con['data']
    tsne = TSNE(random_state=69, n_components=2)
    modelSize = 68000
    testSize = 8000
    count = modelSize*10
    if (count > col.count()*10):
        exit
    data = np.empty((count,11))
    heroesArray = []
    properties = playerOnly.__len__()
    original={'heroNames': [None]*130, 'heroes': [None]*count,'tsne_transformed': [np.zeros(properties+1) for i in repeat(None, 130)]}
    for hero in con['heroes'].find():
        original["heroNames"][hero['id']] = heroesArray.__len__()
        heroesArray.append({'id':hero['id'], 'name':hero['localized_name']})
    k = 0
    finalData = {'data': [np.zeros(properties) for f in repeat(None, heroesArray.__len__())]}
    for x in con["data"].find(limit = modelSize):
        for player in x['players']:
            if player == {} or player['hero_id'] is None:
                continue
            heroId = player['hero_id']
            data[k] = np.fromiter(transform(player, playerOnly).values(), dtype=float)
            original['heroes'][k] = {'id':original['heroNames'][heroId]}
            k+=1
    original['heroes'] = original['heroes'][0:k]
    data = data[0:k]

    print('data is prepared.')
    print('applying tsne.')
    original['data'] = data
    original['tsne'] = data#tsne.fit_transform(original["data"])
    print('tsne is finished.')
    res = original['tsne']
    for i in range(len(res)):
        heroId = original['heroes'][i]['id']
        if not (heroId is None):
            entry = original['tsne_transformed'][heroId]
            for k in range(properties):
                entry[k] += res[i, k]
            entry[properties]+=1

    plt.figure(figsize=(10, 10))
    res = original['tsne_transformed']
    for i in range(len(res)):
        entry = res[i]
        if entry[properties] != 0:
            for k in range(properties):
                finalData['data'][i][k] = entry[k] / entry[properties]
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    res = finalData['data']
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
    clusters = 6
    agg = AgglomerativeClustering(n_clusters=clusters, linkage='ward')
    clusters = agg.fit_predict(res)
    colors = [ "red", "green", "blue", "purple", "white", "pink", "yellow", "orange", "brown", "grey", "blue", "orange", "brown", "grey", "blue", "orange", "brown", "grey", "blue"  ]
    group = []
    for c in colors:
        group.append([])
    for i in range(len(res)):
        plt.text(res[i][0], res[i][1], heroesArray[i]['name'], color=colors[clusters[i]])
        group[clusters[i]].append(heroesArray[i]['name'])
    plt.xlabel("t-SNE feature 0")
    plt.xlabel("t-SNE feature 1")
    for i in range(len(group)):
        print('%d.' % i, end = ' ')
        for hero in group[i]:
            print(hero, end = ', ')
        print('\n')
    model = {}
    k = 0
    for x in con["data"].find(limit = modelSize):
        if any(player is {} or 'hero_id' not in player or player['hero_id'] is None for player in x['players']):
            continue
        key = []
        for player in x['players']:
            heroId = player['hero_id']
            key.append(clusters[original['heroNames'][heroId]])
        radiant = sorted(key[0:5])
        dire = sorted(key[5:10])
        rd = repr(radiant+dire)
        dr = repr(dire+radiant)
        key = rd
        if dr in model:
            key = dr
        elif not (key in model):
            model[key] = {'matches' : 0, 'radiantwin': 0}
        model[key]['matches']+=1
        model[key]['radiantwin']+=int(x['radiant_win'])
        

    testResult = {'matches': 0, 'correct': 0, 'nodata': 0}
    for x in con["data"].find(skip=modelSize, limit = testSize):
        if any(player is {} or 'hero_id' not in player or player['hero_id'] is None for player in x['players']):
            continue
        key = []
        for player in x['players']:
            heroId = player['hero_id']
            key.append(clusters[original['heroNames'][heroId]])
        radiant = sorted(key[0:5])
        dire = sorted(key[5:10])
        rd = repr(radiant+dire)
        dr = repr(dire+radiant)
        if dr in model:
            key = dr
        if rd in model:
            key = rd
        else:
            testResult['nodata']+=1
            continue
        isRadiant = model[key]['radiantwin']*2 >= model[key]['matches']
        testResult['matches']+=1
        testResult['correct']+=int(isRadiant == x['radiant_win'])

    print('No data: %s' % testResult['nodata'])
    print('Matches tested: %s' % testResult['matches'])
    print('Correct predictions: %s' % testResult['correct'])
    print('Precision: %s' % (testResult['correct']/testResult['matches']))

train_hero()
