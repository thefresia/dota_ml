#%%
#from sklearn.manifold import TSNE
#from sklearn.datasets.samples_generator import make_blobs
from db.connection import MongoDbConnection
import scipy as sp
import scipy.interpolate
import time
import enlighten
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import repeat
import numpy as np
import os
import mglearn
import pickle
from pprint import pprint
import logging

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
        data = 0 if not prop in obj else obj[prop]
        if (prop == "players"):
            for i in range(len(data)):
                data[i] = {} if 'leaver_status' in data[i] and (data[i]['leaver_status'] != 0 or data[i]['abandons'] !=0) else transform(data[i], playerProps) 
        newObj[prop] = data
    return newObj

class ModelTraining:
    con = MongoDbConnection.getCollection('dota_ml')
    heroesArray = []
    originalHeroes = [None]*130
    modelSize = 0
    clustersSize = 0
    trainSize = 0
    trainSkip = 0
    averages = []
    activator = 0.5
    multiplier = 1
    matrix = None
    logger = None

    def __init__(self, trainSkip, trainSize, logger = None):
        self.trainSize = trainSize
        self.trainSkip = trainSkip
        self.logger = logger or logging.getLogger(__name__)
        self.prepare_data(trainSkip)
        for hero in self.con['heroes'].find():
            self.originalHeroes[hero['id']] = self.heroesArray.__len__()
            self.heroesArray.append({'id':hero['id'], 'name':hero['localized_name']})
        self.logger.debug(f'Initialized model instance: trainSkip = {trainSkip}; trainSize = {trainSize}.')

    def train_test(self, size, clusters):
        self.prepare_data(size)
        return self.clusterize_train(clusters)
    
    def prepare_data(self, size):
        self.logger.debug(f'Preparing matrix/model for the set of {size} matches.')
        self.modelSize = size
        self._hero_matrix_()
        path = os.path.join(os.path.dirname(__file__), '{}/{}.pickle'.format('averages',size))
        if os.path.exists(path):
            self.logger.debug(f'Found cached data: AVERAGES/{size} - {path}')
            self.averages  = self.___deserialize___(path)
        else:   
            self.logger.debug(f'Evaluating AVERAGES: {size} matches.')
            data = self.___prepare_data___()
            self.averages  = self.___find_averages___(data)
            self.___serialize___('averages',self.modelSize, self.averages)

    def clusterize_train(self, clusters):
        self.clustersSize = clusters
        clustersArray = self.___clusterize___(self.averages)
        model = self.___train_model___(clustersArray)
        return self.___test_model___(clustersArray, model)

    def set_params(self, activator, multiplier):
        self.activator = activator
        self.multiplier = multiplier
        self.logger.debug(f'Set: ACTIVATOR: {activator}, MULTIPLIER: {multiplier}.')

    def ___deserialize___(self, path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def ___serialize___(self, folder, fn, data):
        path = os.path.join(os.path.dirname(__file__), '{}/{}.pickle'.format(folder,fn))
        self.logger.debug(f'Caching file: {path}.')
        with open(path, 'wb+') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _hero_matrix_(self):
        path = os.path.join(os.path.dirname(__file__), '{}/{}.pickle'.format('matrices',self.trainSkip))
        if os.path.exists(path):
            self.logger.debug(f'Found cached data: HEROMATRIX: {self.trainSkip} - {path}')
            self.matrix  = self.___deserialize___(path)
        else:   
            self.logger.debug(f'Building HEROMATRIX: {self.trainSkip} matches.')
            pbar = enlighten.Counter(total=self.trainSkip, desc='Heromatrix', unit='matches')
            matrix = [[{'matches': 0, 'won':0, 'winrate' : 0} for x in range(len(self.heroesArray))] for y in range(len(self.heroesArray))] 
            for x in self.con["data"].find(limit = self.trainSkip):
                if any(player is {} or 'hero_id' not in player or player['hero_id'] in [None,0] for player in x['players']):
                    pbar.update()
                    continue
                for i in range(0,5):
                    for j in range (i+1, 5):
                        self.___eval_matrix_match___(matrix, x, i, j, False)
                    for j in range (5,10):
                        a,b = self.originalHeroes[x['players'][i]['hero_id']], self.originalHeroes[x['players'][j]['hero_id']]
                        r,c = max(a,b), min(a,b)
                        matrix[r][c]['matches']+=1
                        matrix[r][c]['won']+= (a == r) == x['radiant_win']
                for i in range(5,9):
                    for j in range (i+1, 10):
                        self.___eval_matrix_match___(matrix, x, i, j, True)
                pbar.update()
            for i in range(0,len(self.heroesArray)):
                for j in range (i+1, len(self.heroesArray)):
                    matrix[i][j]['winrate'] = 0 if matrix[i][j]['matches'] == 0 else matrix[i][j]['won'] / matrix[i][j]['matches']
                    matrix[j][i]['winrate'] = 0 if matrix[j][i]['matches'] == 0 else matrix[j][i]['won'] / matrix[j][i]['matches']
            self.___serialize___('matrices', self.trainSkip, matrix)
            self.matrix = matrix
            self.logger.debug(f'Done building HEROMATRIX: {self.trainSkip} matches.')
        return self.matrix

    def ___eval_matrix_match___(self, matrix, match, i, j, flag):
        r,c = self.originalHeroes[match['players'][i]['hero_id']], self.originalHeroes[match['players'][j]['hero_id']]
        r,c = min(r,c), max(r,c)
        matrix[r][c]['matches']+=1
        matrix[r][c]['won']+= not match['radiant_win'] if flag else match['radiant_win']

    def ___prepare_data___(self):
        logmsg = 'reading db and transforming matches.'
        self.logger.debug(f'Starting: {logmsg}')
        col = self.con['data']
        count = self.modelSize*10
        if (count > col.count()*10):
            self.logger.critical(f'Not enough data: {col.count()*10}. Exiting the application.')
            exit
        properties = playerOnly.__len__()
        data = {'values' : np.empty((count, properties)), 'heroes' : [None]*count}
        k = 0
        for x in col.find(limit = self.modelSize):
            for player in x['players']:
                if player == {} or player['hero_id'] is None:
                    continue
                heroId = player['hero_id']
                data['values'][k] = np.fromiter(transform(player, playerOnly).values(), dtype=float)
                data['heroes'][k] = self.originalHeroes[heroId]
                k+=1
        data['heroes'] = data['heroes'][0:k]
        data['values'] = data['values'][0:k]
        self.logger.debug(f'Done: {logmsg}')
        return data

    def ___find_averages___(self, data):
        logmsg = 'evaluating average hero statistics.'
        self.logger.debug(f'Starting: {logmsg}')
        pbar = enlighten.Counter(total=len(data['values']), desc='Averages', unit='hero')
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
            pbar.update()
        for i in range(len(averages)):
            entry = averages[i]
            if entry['amount'] != 0:
                for k in range(properties):
                    finalData[i][k] = entry['data'][k] / entry['amount']
        self.logger.debug(f'Done: {logmsg}')
        return finalData
    
    def ___clusterize___(self, averages):
        self.logger.debug(f'Clustering')
        agg = AgglomerativeClustering(n_clusters=self.clustersSize, linkage='ward')
        res = agg.fit_predict(averages)
        self.logger.debug(f'Done clustering.')
        return res
    
    def ___train_model___(self, clusters):
        self.logger.debug(f'Training model: modelSize = {self.modelSize}, clusters = {self.clustersSize}.')
        path = os.path.join(os.path.dirname(__file__), '{}/{}_{}.pickle'.format('models',self.modelSize, self.clustersSize))
        if os.path.exists(path):
            self.logger.debug(f'Found cached data: MODEL/{self.modelSize}_{self.clustersSize} - {path}')
            return self.___deserialize___(path)
        else:   
            pbar = enlighten.Counter(total=self.modelSize, desc='Training', unit='matches')
            model = {}
            for x in self.con["data"].find(limit = self.modelSize):
                if any(player is {} or 'hero_id' not in player or player['hero_id'] in [0,None] for player in x['players']):
                    pbar.update()
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
                pbar.update()
            self.___serialize___('models','{}_{}'.format(self.modelSize, self.clustersSize), model)
            self.logger.debug(f'Done training model: modelSize = {self.modelSize}, clusters = {self.clustersSize}.')
            return model
    
    def ___test_model___(self, clusters, model):
        self.logger.debug(f'Testing model: modelSize = {self.modelSize}, clusters = {self.clustersSize}, tskip|tsize = {self.trainSkip}|{self.trainSize}')
        pbar = enlighten.Counter(total=self.trainSize, desc='Tested', unit='matches')
        testResult = {'matches': 0, 'correct': 0, 'nodata': 0}
        for x in self.con["data"].find(skip=self.trainSkip, limit = self.trainSize):
            if any(player is {} or 'hero_id' not in player or player['hero_id'] in [0,None] for player in x['players']):
                pbar.update()
                continue
            key = []
            hkey = []
            for player in x['players']:
                heroId = player['hero_id']
                hkey.append(self.originalHeroes[heroId])
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
                pbar.update()
                continue
            adv = self.___evaluate_advantage___(hkey)
            evaluation =  model[key]['radiantwin'] / model[key]['matches'] + adv*self.multiplier
            isRadiant = evaluation > self.activator
            testResult['matches']+=1
            testResult['correct']+=int(isRadiant == x['radiant_win'])
            pbar.update()
        self.logger.debug(f'Testing is done!')
        self.logger.debug(f'Accuracy: {testResult["correct"]/testResult["matches"]*100}; correct|tested: {testResult["correct"]}|{testResult["matches"]}; activator|mult: {self.activator}|{self.multiplier}; modelsize|cl = {self.modelSize}|{self.clustersSize}; tskip|tsize = {self.trainSkip}|{self.trainSize}.')
        return testResult

    
    def ___evaluate_advantage___(self, heroes):
        ladv = 0
        radv = 0
        k = 0
        v = 0
        for i in range(0,5):
            for j in range (i+1, 5):
                r,c = min(heroes[i], heroes[j]), max(heroes[i], heroes[j])
                ladv += self.matrix[r][c]['winrate']
            for j in range (5,10):
                r,c = max(heroes[i], heroes[j]), min(heroes[i], heroes[j])
                k+=1
                v += self.matrix[r][c]['winrate'] if heroes[i]>heroes[j] else 1-self.matrix[r][c]['winrate'] 
        for i in range(5,10):
            for j in range (i+1, 10):
                r,c = min(heroes[i], heroes[j]), max(heroes[i], heroes[j])
                radv += self.matrix[r][c]['winrate']
        result = (ladv - radv)/10 + v/25-0.5
        return result

    #print('No data: %s' % testResult['nodata'])
    #print('Matches tested: %s' % testResult['matches'])
    #print('Correct predictions: %s' % testResult['correct'])
    #print('Precision: %s' % (testResult['correct']/testResult['matches']))

#%%
'''from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

m = ModelTraining(70000,6000)
data = lambda x : x['correct']/x['matches']*100
m._hero_matrix_()
m.prepare_data(70000)
m.set_params(0.52,1)
res = m.clusterize_train(2)
print(data(res))

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
activator = np.arange(0.4, 0.6, 0.03, dtype=np.dtype(float))
multiplier = np.arange(1.0, 2.0, 0.1, dtype=np.dtype(float))
X, Y = np.meshgrid(activator, multiplier)
Z = np.copy(X)
for i in range(np.size(Z,1)):
    for j in range(np.size(Z,0)):
        m.set_params(X[0,i].item(), Y[j,i].item())
        trained = m.clusterize_train(2)
        Z[j,i] = data(trained)


surf = ax.plot_surface(X, Y, Z, cmap= cm.get_cmap("coolwarm"), rstride=1, cstride=1,
                       linewidth=0, antialiased=True)

ax.set_zlim(46, 61)

ax.set_xlabel('Активатор') 
ax.set_xticks(activator)
ax.set_ylabel('Множитель') 
ax.set_yticks(multiplier)
ax.set_zlabel('Точность предсказания') 
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.3, aspect=5)

plt.show()

'''
#%%
