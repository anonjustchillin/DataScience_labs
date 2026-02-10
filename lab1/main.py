import pandas as pd
import matplotlib.pyplot as plt
import math as mt
from bs4 import BeautifulSoup as bs
import requests

URL = 'https://www.worldometers.info/world-population/ukraine-population/'
FILENAME = 'data.txt'

HEADER = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
        }


def parse_site(url, filename):
    r = requests.get(url)
    soup = bs(r.content, 'html.parser')
    print(soup)

    info = soup.find('table', class_='datatable')

    with open(filename, "w", encoding='utf-8') as output_file:
        for i in info:
            print(i.text)
            output_file.write(i.text)
            output_file.write('\n')


    return


def get_table(url, filename):
    r = requests.get(url, headers=HEADER)

    dfs = pd.read_html(r.text)
    df = dfs[0]

    #df2 = df[['Year',
    #          'Population',
    #          'Yearly_Percent_Change',
    #          'Yearly_Change',
    #          'Migrants_net',
    #          'Median_Age',
    #          'Fertility_Rate',
    #          'Density',
    #          'Urban_Pop_Percent',
    #          'Urban_Population',
    #          "Share_of_World_Pop",
    #          'World_Population',
    #          'Ukraine_Global_Rank']]
    print(df)


    return


if __name__ == '__main__':
    print(f'URL: {URL}')
    get_table(URL, FILENAME)