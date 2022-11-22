import warnings

import pandas as pd
import re

warnings.filterwarnings('ignore')
csv = pd.read_csv("dataset/dogs.csv")
csv['minweight'] = 0
csv['maxweight'] = 0
csv['minlifespan'] = 0
csv['maxlifespan'] = 0
csv['minsize'] = 0
csv['maxsize'] = 0
for i in range(len(csv['weight'])):
    tmp = csv['weight'][i].split("--")
    csv['minweight'][i] = float(tmp[0])*float(tmp[0])
    csv['maxweight'][i] = float(tmp[1])*float(tmp[1])
for i in range(len(csv['lifespan'])):
    tmp = csv['lifespan'][i].split("--")
    csv['minlifespan'][i] = float(tmp[0])*float(tmp[0])
    csv['maxlifespan'][i] = float(tmp[1])*float(tmp[1])
for i in range(len(csv['size'])):
    tmp = csv['size'][i].split("--")
    csv['minsize'][i] = float(tmp[0])*float(tmp[0])
    csv['maxsize'][i] = float(tmp[1])*float(tmp[1])
csv.to_csv('dataset/finaldogs1.csv')