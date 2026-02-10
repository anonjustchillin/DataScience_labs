import pandas as pd
import matplotlib.pyplot as plt
import math as mt
from bs4 import BeautifulSoup as bs
import requests

URL = 'https://www.worldometers.info/world-population/ukraine-population/'
RAW_FILENAME = 'data.csv'
CLEAN_FILENAME = 'data_cleaned.csv'


HEADER = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
        }

COLUMNS = ['Year',
              'Population',
              'Yearly_Percent_Change',
              'Yearly_Change',
              'Migrants_net',
              'Median_Age',
              'Fertility_Rate',
              'Density',
              'Urban_Pop_Percent',
              'Urban_Population',
              "Share_of_World_Pop",
              'World_Population',
              'Ukraine_Global_Rank']

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

    df.columns = COLUMNS

    df.to_csv(filename)

    return


def view_table(filename):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    print('------- Head (year, population, year%, year_change) -------')
    print(df.loc[:, ['Year',
                    'Population',
                    'Yearly_Percent_Change',
                    'Yearly_Change'
                    ]].head(5))
    print()
    print('------- Tail (year, population, year%, year_change) -------')
    print(df.loc[:, ['Year',
                     'Population',
                     'Yearly_Percent_Change',
                     'Yearly_Change'
                     ]].tail(5))
    print()
    print()

    print('------- Head (migrant_net, med_age, fert_rate, dens, urban%) -------')
    print(df.loc[:, ['Migrants_net',
                    'Median_Age',
                    'Fertility_Rate',
                    'Density',
                    'Urban_Pop_Percent']].head(5))
    print()
    print('------- Tail (migrant_net, med_age, fert_rate, dens, urban%) -------')
    print(df.loc[:, ['Migrants_net',
                     'Median_Age',
                     'Fertility_Rate',
                     'Density',
                     'Urban_Pop_Percent']].tail(5))
    print()
    print()

    print('------- Head (urban_pop, share_world, world_pop, global_rank) -------')
    print(df.loc[:, [
                    'Urban_Population',
                    "Share_of_World_Pop",
                    'World_Population',
                    'Ukraine_Global_Rank']].head(5))
    print()
    print('------- Tail (urban_pop, share_world, world_pop, global_rank) -------')
    print(df.loc[:, [
                        'Urban_Population',
                        "Share_of_World_Pop",
                        'World_Population',
                        'Ukraine_Global_Rank']].tail(5))
    print()
    print()

    print(df.info())
    return


def clean_table(filename, output_filename):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)

    df = df.replace('%', '', regex=True)
    df = df.replace(',', '', regex=True)
    df = df.replace('â', '-', regex=True)

    conv_col = {'Yearly_Change': int,
                'Migrants_net': int}
    df = df.astype(conv_col)

    df.to_csv(output_filename)

    return


if __name__ == '__main__':
    print(f'URL: {URL}')
    #get_table(URL, RAW_FILENAME)

    print('------------------- RAW DATASET -------------------')
    view_table(RAW_FILENAME)
    print('---------------------------------------------------')

    clean_table(RAW_FILENAME, CLEAN_FILENAME)
    print()

    print('------------------- CLEANED DATASET -------------------')
    view_table(CLEAN_FILENAME)
    print('---------------------------------------------------')
