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


URL = 'https://www.worldometers.info/water/'
RAW_FILENAME = 'data_raw.csv'
CLEAN_FILENAME = 'data_cleaned.csv'
PROJECT_PATH = 'D:\\uni\\3курс\Data_Science\Data_science_labs\lab2'

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

# parse site
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

# add anomalies
def add_anomalies(df):
    return

# clean df
def clean_df(df):
    return

# view stats
def view_stats(df):
    return

# MNK
def analyze_MNK(df, interval = 0.5):
    return

# smoothing
def smoothing(df):
    return


if __name__ == '__main__':
    print(f'URL: {URL}')

    DATA_PATH = os.path.join(PROJECT_PATH, RAW_FILENAME)
    CLEANED_PATH = os.path.join(PROJECT_PATH, CLEAN_FILENAME)