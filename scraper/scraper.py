import requests
from bs4 import BeautifulSoup
import time
import json


def extract_data(page): #extracts author, title, text and type(prose or poen) from html. Returns a dictionary.
    parser = BeautifulSoup(page.content, 'html.parser')

    author = parser.find('div', class_='autor').text
    title = parser.find('h1', class_='titulo-texto').text
    
    text_element = parser.find('div', class_='texto-poesia')
    if text_element:
        text = text_element.text
        text_type = "poem"
    else:
        text = parser.find('div', class_='texto-prosa').text
        text_type = "prose"


    data = {
        'author': author,
        'title': title,
        'text': text,
        'text_type': text_type
    }

    return data



data = [] #list to save the data

i = 0
max_id = 4545 #seems that 4544 is the last text
while i < max_id: #extracting loop
    url = f'http://arquivopessoa.net/textos/{i}'
    page = requests.get(url)
    if page.status_code == 200:
        print(f'Extracting data from {i}')
        data.append(extract_data(page))
        i += 1

    elif page.status_code == 503: #if error accessing the page retries after t seconds.
        t = 2
        print(f'Error 503 retrying to fetch page {i} in {t} seconds')
        time.sleep(t)
        
    else:
        print(f'No {i} : Status {page.status_code}')
        i += 1

with open('texts.json', 'w') as texts: #writes data into a json file
    json.dump(data, texts)
