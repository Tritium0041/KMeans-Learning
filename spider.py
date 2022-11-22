import re
import time

import requests
import pandas as pd
req = requests.get("https://www.bestpuppy.com/dog-breeds").text
dogNames = re.findall("(?<=<img alt=\").*?(?=\" src)", req)
dogNames.pop()
for i in range(5):
    del(dogNames[0])
for i in range(0, len(dogNames)):
    dogNames[i] = dogNames[i].lower()
    dogNames[i] = dogNames[i].replace(" ", "-")
print(dogNames)#狗名列表获取结束
data = []
for i in range(0, len(dogNames)):
    req = requests.get(f"https://www.bestpuppy.com/dog-breeds/{dogNames[i]}").text
    res =re.findall("(?<=<span class=\"s-16 d-block line-height-10 mt-5\">).*?(?=</span>)", req)
    res.append(dogNames[i])
    data.append(res)
    print(i)
    time.sleep(0.1)
csv = pd.DataFrame(data,columns=['weight','country','lifespan','size',"breed"])
csv.to_csv("dataset/dogs.csv")