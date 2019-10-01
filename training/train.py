#%%
#from sklearn.manifold import TSNE
#from sklearn.datasets.samples_generator import make_blobs
import scipy as sp
import scipy.interpolate
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import repeat
import numpy as np
import os
import pymongo
import mglearn
import pickle
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
    trainSkip = 0
    averages = []
    matrix = None

    def __init__(self, trainSkip, trainSize):
        self.trainSize = trainSize
        self.trainSkip = trainSkip
        for hero in self.con['heroes'].find():
            self.originalHeroes[hero['id']] = self.heroesArray.__len__()
            self.heroesArray.append({'id':hero['id'], 'name':hero['localized_name']})

    def train_test(self, size, clusters):
        self.prepare_data(size)
        return self.clusterize_train(clusters)
    
    def prepare_data(self, size):
        self.modelSize = size
        path = os.path.join(os.path.dirname(__file__), '{}/{}.pickle'.format('averages',size))
        if os.path.exists(path):
            self.averages  = self.___deserialize___(path)
        else:   
            data = self.___prepare_data___()
            self.averages  = self.___find_averages___(data)
            self.___serialize___('averages',self.modelSize, self.averages)


    def clusterize_train(self, clusters):
        self.clustersSize = clusters
        clustersArray = self.___clusterize___(self.averages)
        model = self.___train_model___(clustersArray)
        return self.___test_model___(clustersArray, model)

    def ___deserialize___(self, path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def ___serialize___(self, folder, fn, data):
        path = os.path.join(os.path.dirname(__file__), '{}/{}.pickle'.format(folder,fn))
        print('Saved: %s'%path)
        with open(path, 'wb+') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _hero_matrix_(self):
        path = os.path.join(os.path.dirname(__file__), '{}/{}.pickle'.format('matrices',self.trainSkip))
        if os.path.exists(path):
            self.matrix  = self.___deserialize___(path)
        else:   
            matrix = [[{'matches': 0, 'won':0, 'winrate' : 0} for x in range(len(self.heroesArray))] for y in range(len(self.heroesArray))] 
            for x in self.con["data"].find(limit = self.trainSkip):
                if any(player is {} or 'hero_id' not in player or player['hero_id'] is None for player in x['players']):
                    continue
                for i in range(0,4):
                    for j in range (i+1, 4):
                        if (i == j): 
                            break
                        r,c = self.originalHeroes[x['players'][i]['hero_id']], self.originalHeroes[x['players'][j]['hero_id']]
                        r,c = min(r,c), max(r,c)
                        matrix[r][c]['matches']+=1
                        matrix[r][c]['won']+=x['radiant_win']
                    for j in range (5,9):
                        a,b = self.originalHeroes[x['players'][i]['hero_id']], self.originalHeroes[x['players'][j]['hero_id']]
                        r,c = max(a,b), min(a,b)
                        matrix[r][c]['matches']+=1
                        matrix[r][c]['won']+= (a == r) == x['radiant_win']
                for i in range(5,8):
                    for j in range (i+1, 9):
                        r,c = self.originalHeroes[x['players'][i]['hero_id']], self.originalHeroes[x['players'][j]['hero_id']]
                        r,c = min(r,c), max(r,c)
                        matrix[r][c]['matches']+=1
                        matrix[r][c]['won']+= not x['radiant_win']
            for i in range(0,len(self.heroesArray)):
                for j in range (0, len(self.heroesArray)):
                    if (i == j):
                        continue
                    matrix[i][j]['winrate'] = 0 if matrix[i][j]['matches'] == 0 else matrix[i][j]['won'] / matrix[i][j]['matches']
            self.___serialize___('matrices', self.trainSkip, matrix)
            self.matrix = matrix
        return self.matrix




    def ___prepare_data___(self):
        con = MongoDbConnection.getCollection("dota_ml")
        col = self.con['data']
        count = self.modelSize*10
        if (count > col.count()*10):
            print('Not enough data: %d'%col.count()*10)
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

    def ___find_averages___(self, data):
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
    
    def ___clusterize___(self, averages):
        agg = AgglomerativeClustering(n_clusters=self.clustersSize, linkage='ward')
        return agg.fit_predict(averages)
    
    def ___train_model___(self, clusters):
        path = os.path.join(os.path.dirname(__file__), '{}/{}_{}.pickle'.format('models',self.modelSize, self.clustersSize))
        if os.path.exists(path):
            return self.___deserialize___(path)
        else:   
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
            self.___serialize___('models','{}_{}'.format(self.modelSize, self.clustersSize), model)
            return model
    
    def ___test_model___(self, clusters, model):
        testResult = {'matches': 0, 'correct': 0, 'nodata': 0}
        for x in self.con["data"].find(skip=self.trainSkip, limit = self.trainSize):
            if any(player is {} or 'hero_id' not in player or player['hero_id'] is None for player in x['players']):
                continue
            key = []
            rkey = []
            for player in x['players']:
                heroId = player['hero_id']
                key.append(clusters[self.originalHeroes[heroId]])
            radiant = sorted(key[0:5])
            dire = sorted(key[5:10])
            rd = repr(radiant+dire)
            dr = repr(dire+radiant)
            if dr in model:
                rkey = dr
            elif rd in model:
                rkey = rd
            else:
                testResult['nodata']+=1
                continue
            evaluation = model[rkey]['radiantwin'] / model[rkey]['matches'] + self.___evaluate_advantage___(key)
            print(str(self.___evaluate_advantage___(key)) + ' ' + str(model[rkey]['radiantwin'] / model[rkey]['matches']) + ' ' + str(evaluation) + ' ' + str(x['radiant_win']))
            isRadiant = evaluation > 0.5487
            testResult['matches']+=1
            testResult['correct']+=int(isRadiant == x['radiant_win'])
        return testResult
    
    def ___evaluate_advantage___(self, heroes):
        ladv = 0
        radv = 0
        v = 0
        for i in range(0,4):
            for j in range (i+1, 4):
                if (i == j): 
                    break
                r,c = min(heroes[i], heroes[j]), max(heroes[i], heroes[j])
                ladv += self.matrix[r][c]['winrate']
            for j in range (5,9):
                r,c = max(heroes[i], heroes[j]), min(heroes[i], heroes[j])
                v += self.matrix[r][c]['winrate']
        for i in range(5,8):
            for j in range (i+1, 9):
                r,c = min(heroes[i], heroes[j]), max(heroes[i], heroes[j])
                radv += self.matrix[r][c]['winrate']
        result = (ladv - radv)/15 + v/250 
        return result 

    #print('No data: %s' % testResult['nodata'])
    #print('Matches tested: %s' % testResult['matches'])
    #print('Correct predictions: %s' % testResult['correct'])
    #print('Precision: %s' % (testResult['correct']/testResult['matches']))

#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

m = ModelTraining(93000,6000)
matr = m._hero_matrix_()
m.prepare_data(90000)
trained = m.clusterize_train(4)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
samples = np.arange(30000, 93000, 3000, dtype=np.dtype(int))
clusters = np.arange(3, 10, 1, dtype=np.dtype(int))
X, Y = np.meshgrid(samples, clusters)
data = lambda x : x['correct']/x['matches']*100
Z = np.copy(X)
for i in range(np.size(Z,1)):
    m.prepare_data(X[0,i].item())
    for j in range(np.size(Z,0)):
        trained = m.clusterize_train(Y[j,i].item())
        Z[j,i] = data(trained)


surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rstride=1, cstride=1,
                       linewidth=0, antialiased=True)

ax.set_zlim(48, 56)

ax.set_xlabel('Обучающая выборка') 
ax.set_xticks(samples)
ax.set_ylabel('Количество кластеров') 
ax.set_yticks(clusters)
ax.set_zlabel('Точность предсказания') 
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.3, aspect=5)

plt.show()


#%%
