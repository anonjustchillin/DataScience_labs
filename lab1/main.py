import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math as mt
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import os.path

URL = 'https://www.worldometers.info/world-population/ukraine-population/'
RAW_FILENAME = 'data.csv'
CLEAN_FILENAME = 'data_cleaned.csv'
PROJECT_PATH = 'D:\\uni\\3курс\Data_Science\Data_science_labs\lab1'

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
    print(df.loc[:, [
                    'Population',
                    'Yearly_Percent_Change',
                    'Yearly_Change'
                    ]].head(5))
    print()
    print('------- Tail (year, population, year%, year_change) -------')
    print(df.loc[:, [
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
    df.Year = pd.to_datetime(df.Year, format='%Y')
    df.set_index('Year', inplace=True)

    df = df[::-1]

    df.to_csv(output_filename)

    return


def plot_stuff(filename):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)

    for col in df.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df[col])
        plt.title(col)
        plt.xlabel('Year')
        plt.ylabel(col)
        plt.xticks(rotation=30, ha='right')
        plt.grid(True)
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.show()

    return


def stats(filename, chosen_col):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)

    #print(df[chosen_col].describe())

    col = df[chosen_col].tolist()
    print(col)
    print()

    mean_col = round(np.mean(col), 4)
    med_col = round(np.median(col), 4)
    var_col = round(np.var(col), 4)
    std_col = round(np.std(col), 4)
    sqrt_col = round(mt.sqrt(var_col), 4)

    print(f'Length = {len(col)}')
    print(f'Mean = {mean_col}')
    print(f'Median = {med_col}')
    print(f'Variance = {var_col}')
    print(f'Standard deviation = {std_col}')
    print(f'Sqrt = {sqrt_col}')

    plt.hist(col)
    plt.grid(True)
    plt.title(f'Histogram of {chosen_col}')
    plt.show()

    return


'''
Закон зміни похибки – експонентційний, нормальний;
Закон зміни досліджуваного процесу (тренду) – лінійний, квадратичний.
Комбінаторика похибка / тренд – довільна.
Реальні дані – 3 показники.
'''


if __name__ == '__main__':
    print(f'URL: {URL}')

    DATA_PATH = os.path.join(PROJECT_PATH, RAW_FILENAME)
    CLEANED_PATH = os.path.join(PROJECT_PATH, CLEAN_FILENAME)

    if not os.path.exists(DATA_PATH):
        get_table(URL, RAW_FILENAME)
        print('------------------- RAW DATASET -------------------')
        view_table(RAW_FILENAME)
        print('---------------------------------------------------')
        print()
    if not os.path.exists(CLEANED_PATH):
        clean_table(RAW_FILENAME, CLEAN_FILENAME)
        print('------------------- CLEANED DATASET -------------------')
        view_table(CLEAN_FILENAME)
        print('---------------------------------------------------')
        print()

    #plot_stuff(CLEAN_FILENAME)

    stats(CLEAN_FILENAME, 'Population')
