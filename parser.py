from bs4 import BeautifulSoup
import requests
import numpy as np

headers = {
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36'
    }

def parse_heroes(filePath):
    url = 'https://ru.dotabuff.com/heroes'
    r = requests.get(url, headers = headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    grid = soup.find("div", class_= "hero-grid")
    f = open(filePath, "wb")
    for a in grid.find_all('a'):
        href = a.get('href')
        hero = href[href.rfind('/')+1:]
        print(hero)
        f.write((hero+'\n').encode('utf-8'))
    f.close()

def parse_winrates():
    with open('heroes.txt') as f:
        heroes = f.readlines()
        i = 1
        amount = str(heroes.__len__())
        for hero in heroes:
            hero = hero.strip()
            print(str(i)+'/'+amount+' :'+hero)
            i+=1
            url = 'https://ru.dotabuff.com/heroes/'+hero+'/counters?date=patch_7.21'
            r = requests.get(url, headers = headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', class_= "sortable").find('tbody')
            data = []
            elements = list(table.find_all('tr'))
            for e in elements:
                enemy = str(e.get('data-link-to'))
                enemy = enemy[enemy.rfind('/')+1:]
                tags = list(e.find_all('td'))
                if len(tags) == 5:
                    data.append([enemy, tags[2].get('data-value'), 
                    tags[3].get('data-value'),tags[4].get('data-value')])
            data.append([hero, '-1','-1','-1'])
            data.sort(key = lambda x: x[0])
            with open('data/'+hero+'.txt', "w") as f:
                for d in data:
                    f.write(d[1]+','+d[2]+','+d[3]+'\n')

            
parse_winrates()
print('ok')
    