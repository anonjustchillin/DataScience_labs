import pandas as pd
import matplotlib.pyplot as plt
import math as mt
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import os.path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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


def plot_stuff(col, col_name):
    plt.figure(figsize=(15, 6))
    plt.plot(col)
    plt.title(col_name)
    plt.xlabel('Year')
    plt.ylabel(col_name)
    plt.xticks(rotation=30, ha='right')
    plt.grid(True)
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.show()

    return


def stats(col):
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
    return


def show_hist(col, col_name):
    plt.hist(col, density=True, edgecolor='black')
    plt.grid(True)
    plt.title(f'Histogram of {col_name}')
    plt.show()
    return


def arima_params(col):
    plot_acf(col)
    plot_pacf(col)
    plt.show()
    return


def check_stationarity(col):
    result = adfuller(col.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("Stationary\n")
        return True
    else:
        print("Non-stationary\n")
        return False


def arima_forecast(col, col_name):
    is_stat = check_stationarity(col)
    arima_params(col)
    if not is_stat:
        col2 = col.diff().fillna(0)
        is_stat = check_stationarity(col2)
        plot_stuff(col2, col_name)
        show_hist(col2.to_list(), col_name)
        arima_params(col2)

    p, d, q = int(input("p = ")), int(input("d = ")), int(input("q = "))
    step = 10
    n = len(col)

    model = ARIMA(col, order=(p, d, q))
    model_fit = model.fit()
    pred = model_fit.predict(start=n-1, end=n+step)

    plt.figure(figsize=(15, 6))
    plt.plot(col.to_list(), label='Actual data')
    plt.plot(pred, color='red', label='Forecasted data')
    plt.title(f'ARIMA forecast {col_name}')
    plt.ylabel(col_name)
    plt.grid(True)
    plt.legend()
    plt.show()

    return


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

    df = pd.read_csv(CLEAN_FILENAME, encoding='utf-8', index_col=0)

    #end_loop = len(COLUMNS)-1
    end_loop = 2
    for col in COLUMNS[1:end_loop]:
        print(f'--------------------------{col}---------------------------')
        selected_col_year = df[col].copy()

        plot_stuff(selected_col_year, col)

        selected_col = selected_col_year.to_list()

        show_hist(selected_col, col)

        stats(selected_col)
        print()

        arima_forecast(selected_col_year, col)

        print('-----------------------------------------------------')
        print()
