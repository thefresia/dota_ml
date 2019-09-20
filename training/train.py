#%%
from sklearn.manifold import TSNE
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    "gold_per_min","hero_damage","hero_healing",
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

class ModelTraining:
    con = MongoDbConnection.getCollection("dota_ml")
    heroesArray = []
    originalHeroes = [None]*130
    modelSize = 0
    clustersSize = 0
    trainSize = 0
    averages = []

    def __init__(self, trainSize):
        self.trainSize = trainSize
        for hero in self.con['heroes'].find():
            self.originalHeroes[hero['id']] = self.heroesArray.__len__()
            self.heroesArray.append({'id':hero['id'], 'name':hero['localized_name']})

    def train_test(self, size, clusters):
        self.modelSize = size
        self.clustersSize = clusters
        data = self.prepareModel()
        averages = self.findAverages(data)
        self.averages = averages
        clustersArray = self.clusterize(averages)
        model = self.trainModel(clustersArray)
        return self.testModel(clustersArray, model)
    
    def prepare_data(self, size):
        self.modelSize = size
        data = self.prepareModel()
        self.averages  = self.findAverages(data)
    
    def clusterize_train(self, clusters):
        self.clustersSize = clusters
        clustersArray = self.clusterize(self.averages)
        model = self.trainModel(clustersArray)
        return self.testModel(clustersArray, model)

    def prepareModel(self):
        con = MongoDbConnection.getCollection("dota_ml")
        col = self.con['data']
        count = self.modelSize*10
        if (count > col.count()*10):
            exit
        properties = playerOnly.__len__()
        data = {'values' : np.empty((count, properties)), 'heroes' : [None]*count}
        k = 0
        for x in con["data"].find(limit = self.modelSize):
            for player in x['players']:
                if player == {} or player['hero_id'] is None:
                    continue
                heroId = player['hero_id']
                data['values'][k] = np.fromiter(transform(player, playerOnly).values(), dtype=float)
                data['heroes'][k] = self.originalHeroes[heroId]
                k+=1
        data['heroes'] = data['heroes'][0:k]
        data['values'] = data['values'][0:k]
        return data

    def findAverages(self, data):
        properties = playerOnly.__len__()
        averages = [{ 'data': np.zeros(properties), 'amount' : 0 } for f in repeat(None, self.heroesArray.__len__())]
        finalData = [np.zeros(properties) for f in repeat(None, self.heroesArray.__len__())]
        for i in range(len(data['values'])):
            heroId = data['heroes'][i]
            if not (heroId is None):
                entry = averages[heroId]
                for k in range(properties):
                    entry['data'][k] += data['values'][i, k]
                entry['amount']+=1
        for i in range(len(averages)):
            entry = averages[i]
            if entry['amount'] != 0:
                for k in range(properties):
                    finalData[i][k] = entry['data'][k] / entry['amount']
        return finalData
    
    def clusterize(self, averages):
        agg = AgglomerativeClustering(n_clusters=self.clustersSize, linkage='ward')
        return agg.fit_predict(averages)
    
    def trainModel(self, clusters):
        model = {}
        for x in self.con["data"].find(limit = self.modelSize):
            if any(player is {} or 'hero_id' not in player or player['hero_id'] is None for player in x['players']):
                continue
            key = []
            for player in x['players']:
                heroId = player['hero_id']
                key.append(clusters[self.originalHeroes[heroId]])
            radiant = sorted(key[0:5])
            dire = sorted(key[5:10])
            rd = repr(radiant+dire)
            dr = repr(dire+radiant)
            key = rd
            if dr in model:
                key = dr
            elif rd in model:
                key = rd
            else:
                model[key] = {'matches' : 0, 'radiantwin': 0}
            model[key]['matches']+=1
            model[key]['radiantwin']+=int(x['radiant_win'])
        return model
    
    def testModel(self, clusters, model):
        testResult = {'matches': 0, 'correct': 0, 'nodata': 0}
        for x in self.con["data"].find(skip=self.modelSize, limit = self.trainSize):
            if any(player is {} or 'hero_id' not in player or player['hero_id'] is None for player in x['players']):
                continue
            key = []
            for player in x['players']:
                heroId = player['hero_id']
                key.append(clusters[self.originalHeroes[heroId]])
            radiant = sorted(key[0:5])
            dire = sorted(key[5:10])
            rd = repr(radiant+dire)
            dr = repr(dire+radiant)
            if dr in model:
                key = dr
            elif rd in model:
                key = rd
            else:
                testResult['nodata']+=1
                continue
            isRadiant = model[key]['radiantwin']*2 >= model[key]['matches']
            testResult['matches']+=1
            testResult['correct']+=int(isRadiant == x['radiant_win'])
        return testResult

def train_hero():
    con = MongoDbConnection.getCollection("dota_ml")
    col = con['data']
    tsne = TSNE(random_state=69, n_components=2)
    modelSize = 68000
    testSize = 8000
    count = modelSize*10
    if (count > col.count()*10):
        exit
    data = np.empty((count,10))
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
    clusters = 4
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
        elif rd in model:
            key = rd
        else:
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
        elif rd in model:
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

#train_hero()
#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

m = ModelTraining(6000)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
samples = np.arange(30000, 80000, 10000, dtype=np.dtype(int))
clusters = np.arange(4, 9, 1, dtype=np.dtype(int))
samples, clusters = np.meshgrid(samples, clusters)
data = lambda x : x['correct']/x['matches']
Z = np.copy(samples)
for i in range(np.size(Z,0)):
    m.prepare_data(samples[i,i].item())
    for j in range(np.size(Z,1)):
        trained = m.train_test(samples[i,j].item(), clusters[i,j].item(), 6000)
        Z[i,j] = data(trained)


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-10, 10)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


#%%
