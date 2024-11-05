import requests
from bs4 import BeautifulSoup

texts = []

for i in range(5000):
    url = f'http://arquivopessoa.net/textos/{i}'
    arquivo = requests.get(url)
    if arquivo == '<Response [200]>':
        texts.append(i)

#print(arquivo)

#soup = BeautifulSoup(arquivo.content, 'html.parser')
#print(soup.prettify())