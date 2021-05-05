import requests
import json
from bs4 import BeautifulSoup
import pandas as pd

# cookie = '_T_WM=88639692422; XSRF-TOKEN=87747d; WEIBOCN_FROM=1110006030; ALF=1585709939; SCF=AtevLCXcVe8LT4nAVb8g6EltagehsRPyU_jTGJ2qyQKKapd5INSmAOMR6hTTayNH0hMhGtsMWjuR6cWXgBj0MtA.; SUB=_2A25zWAYkDeRhGeFM4lAW-CvNwjmIHXVQoqpsrDV6PUNbktAKLXjakW1NQIiIWpoLhFh7DXU1pQK5LNlytMzgrZwq; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhUc88m0Pwex6PiW8R_Bn635JpX5KMhUgL.FoME1KzN1h-p1K-2dJLoI7DsqgLXIhMpeKef; SUHB=09PtzjmNf9wMmP; SSOLoginState=1583117940; MLOGIN=1; M_WEIBOCN_PARAMS=oid%3D4477942912256461%26luicode%3D20000061%26lfid%3D4477942912256461%26uicode%3D20000061%26fid%3D4477942912256461'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'

cookie = 'M_WEIBOCN_PARAMS=oid%3D4557670105224109%26luicode%3D10000011%26lfid%3D100103type%253D1%2526q%253D%25E5%2585%25B1%25E9%259D%2592%25E5%259B%25A2%25E4%25B8%25AD%25E5%25A4%25AE%26uicode%3D20000061%26fid%3D4557670105224109; expires=Sun, 11-Oct-2020 12:39:00 GMT; Max-Age=600; path=/; domain=.weibo.cn; HttpOnly'

url = 'https://m.weibo.cn/comments/hotflow?id=4477659188731756&mid=4477659188731756&max_id=0'
# url = 'https://m.weibo.cn/comments/hotflow?id=4557670105224109&mid=4557670105224109&max_id=0'
# url = 'https://m.weibo.cn/comments/hotflow?id=4558945202483151&mid=4558945202483151&max_id_type=0'
headers = {
    'Cookie': cookie,
    'User-Agent': user_agent
}


def get_page(max_id, id_type):
    params = {
        'max_id': max_id,
        'max_id_type': id_type
    }
    try:
        r = requests.get(url, params=params, headers=headers)
        # if r.status_code == 200:
        if r.content:
            return r.json()
    except requests.ConnectionError as e:
        print('error', e.args)
        get_page(max_id, id_type)


def text_clean(text):
    clean_text = BeautifulSoup(text, 'lxml').get_text()
    return clean_text


def get_text(r_json):
    try:
        if r_json:
            for messege in r_json.get('data').get('data'):
                text = text_clean(messege.get("text"))
                like = messege.get('like_count')
                texts.append(text)
                likes.append(like)
    except AttributeError:
        return 0

def get_next(r_json):
    items_max_id = 0
    items_max_id_type = 0
    if r_json:
        items = r_json.get('data')
        items_max_id = items['max_id']
        items_max_id_type = items['max_id_type']
    return items_max_id, items_max_id_type

max_id = 0
id_type = 0
texts = []
likes = []
if_continue = 1

for i in range(0, 10):
    if (i+1) % 5 == 0:
        print('Epoch{}/{}'.format(i+1, 100))
    r = get_page(max_id, id_type)
    if_continue = get_text(r)
    if if_continue == 0:
        break
    max_id, id_type = get_next(r)
    print(i)
    print(texts)


df = pd.DataFrame({'text':texts, 'like_count':likes})
df.to_csv("D:\work\comment.csv", encoding="utf_8_sig")