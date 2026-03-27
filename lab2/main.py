import pandas as pd
import matplotlib.pyplot as plt
import math as mt
import numpy as np
import re
from bs4 import BeautifulSoup as bs
import requests
import os.path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


URL = 'https://www.worldometers.info/water/'
RAW_FILENAME = 'data_raw.csv'
CLEAN_FILENAME = 'data_cleaned.csv'
CLEAN_NONA_FILENAME = 'data_cleaned_nona.csv'
PROJECT_PATH = 'D:\\uni\\3курс\Data_Science\Data_science_labs\lab2'

PRINT_SEP = '-------------------------------------'
COL_NAME = 'Water_Withdrawal'

START_DATE = 1901
END_DATE = 2014
SKIPPED_DATES = [2011, 2012, 2013]

HEADER = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
        }


# view site
def view_site(url):
    r = requests.get(url)
    soup = bs(r.content, 'html.parser')
    print(soup)
    return


# view script
def view_script(url):
    r = requests.get(url)
    soup = bs(r.content, 'html.parser')
    info = soup.find_all('script')
    print(info)
    return


# parse site
def parse_site(url, filename=RAW_FILENAME):
    r = requests.get(url)
    soup = bs(r.content, 'html.parser')

    info = soup.find_all('script')
    required_info = re.search(r'const data = \[(.*?)\]', str(info), flags=re.S).group(1)
    #print(required_info)

    str_len = len(required_info)
    #print(f'str_len = {str_len}')

    curr_date = START_DATE
    with open(filename, "w", encoding='utf-8') as output_file:
        output_file.write(f",{COL_NAME}\n")
        output_file.write(str(curr_date))
        output_file.write(",")
        for i in range(str_len):
            curr_line = required_info[i]
            if curr_line == "\n" or curr_line == " ":
                continue
            elif curr_line == ",":
                output_file.write('\n')
                if i < str_len - 10:
                    curr_date += 1
                    while curr_date in SKIPPED_DATES:
                        curr_date += 1
                    output_file.write(str(curr_date))
                    output_file.write(",")
            else:
                output_file.write(curr_line)

    return


# clean df
def clean_df(filename, output_filename=CLEAN_FILENAME):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)

    # remove /1_000_000
    # remove _ in numbers
    rmv_str = "/1_000_000"
    def edit_str(x):
        x = x.replace(rmv_str, '')
        x = x.replace('_', '')
        return x

    df = df[COL_NAME].apply(edit_str)

    # divide by 1 000 000
    dvd_by = 1000000
    df = df.astype(float)
    df = df.apply(lambda x: x/dvd_by)

    # create 2011, 2012, 2013
    for i in range(2011, END_DATE):
        df.loc[i] = pd.NA
    df = df.sort_index()

    print(df)
    df.to_csv(output_filename)
    return


def fill_empty_cells(filename, output_filename=CLEAN_NONA_FILENAME, view_options=False):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    col = df.iloc[:, 0]

    if view_options:
        # simple (ffill, bfill, median)
        df_ffill = df.ffill()
        output_stats_graph(df=df_ffill, comment=f'{COL_NAME} (ffill)', show_end=True)

        df_bfill = df.bfill()
        output_stats_graph(df=df_bfill, comment=f'{COL_NAME} (bfill)', show_end=True)

        col_median = col.median(skipna=True)
        df_median = df.fillna(value=col_median)
        output_stats_graph(df=df_median, comment=f'{COL_NAME} (median)', show_end=True)

        # interpolation (linear, polynomial, spline)
        df_lin = df.interpolate(method='linear')
        output_stats_graph(df=df_lin, comment=f'{COL_NAME} (linear)', show_end=True)

        df_pol = df.interpolate(method='polynomial', order=2)
        output_stats_graph(df=df_pol, comment=f'{COL_NAME} (polynomial 2)', show_end=True)

        df_spl = df.interpolate(method='polynomial', order=3)
        output_stats_graph(df=df_spl, comment=f'{COL_NAME} (polynomial 3)', show_end=True)
    else:
        df = df.interpolate(method='linear')
        df.to_csv(output_filename)

    return


def kalman_filter():
    return


def plot_stuff(col, col_name=COL_NAME, show_end=False):
    start_date = START_DATE
    end_date = END_DATE

    plt.figure(figsize=(15, 6))
    plt.plot(col)
    plt.title(col_name)
    plt.xlabel('Year')
    plt.ylabel(COL_NAME)
    plt.xticks(np.arange(start_date, end_date, 5), rotation=30, ha='right')
    if show_end:
        plt.xlim(end_date-10, end_date)
    plt.grid(True)
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.show()

    return


def show_hist(col, col_name=COL_NAME):
    plt.hist(col, density=True, edgecolor='black')
    plt.grid(True)
    plt.title(f'Histogram of {col_name}')
    plt.show()
    return


def view_stats(col, comment=COL_NAME):
    print(comment)
    mean_col = round(np.mean(col), 4)
    med_col = round(np.median(col), 4)
    if np.isnan(med_col):
        med_col = col.median(skipna=True)
    var_col = round(np.var(col), 4)
    std_col = round(np.std(col), 4)

    print(f'Length = {len(col)}')
    print(f'Mean = {mean_col}')
    print(f'Median = {med_col}')
    print(f'Variance = {var_col}')
    print(f'Standard deviation = {std_col}')

    print(PRINT_SEP)

    return


def check_stationarity(filename, skip_last_values=False, start_idx=0, end_idx=-4):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)

    if skip_last_values:
        col = df.iloc[start_idx:end_idx, 0]
    else:
        col = df.iloc[:, 0]

    def test_stationarity(x):
        result = adfuller(x)

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

    is_stat = test_stationarity(col.values)

    print(PRINT_SEP)
    #while not is_stat:
    #    col2 = col.diff().fillna(0)
    #    is_stat = test_stationarity(col2.values)
    #    plot_stuff(col2, f'{COL_NAME} (diff)')
    #    show_hist(col2, f'{COL_NAME} (diff)')


# MNK
def analyze_MNK(df, interval = 0.5):
    return


# smoothing
def smoothing(df):
    return


def output_stats_graph(filename=None, df=None, col=None, comment=None, show_end=False):
    if filename is not None:
        df = pd.read_csv(filename, encoding='utf-8', index_col=0)
        col = df.iloc[:, 0]
    elif df is not None:
        col = df.iloc[:, 0]

    plot_stuff(col, comment, show_end)
    show_hist(col, comment)
    view_stats(col, comment)


if __name__ == '__main__':
    print(f'URL: {URL}')

    DATA_PATH = os.path.join(PROJECT_PATH, RAW_FILENAME)
    CLEANED_PATH = os.path.join(PROJECT_PATH, CLEAN_FILENAME)
    CLEANED_NONA_PATH = os.path.join(PROJECT_PATH, CLEAN_NONA_FILENAME)

    if not os.path.isfile(DATA_PATH):
        parse_site(URL)
    if not os.path.isfile(CLEANED_PATH):
        clean_df(DATA_PATH)

    #plot_stuff(CLEANED_PATH)
    #show_hist(CLEANED_PATH)
    #view_stats(CLEANED_PATH)
    #print(PRINT_SEP)
    #check_stationarity(CLEANED_PATH, True)
    fill_empty_cells(CLEANED_PATH, output_filename=CLEANED_NONA_PATH)
