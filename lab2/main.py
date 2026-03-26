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
PROJECT_PATH = 'D:\\uni\\3курс\Data_Science\Data_science_labs\lab2'

START_DATE = 1901
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
def parse_site(url, filename='textfile.csv'):
    r = requests.get(url)
    soup = bs(r.content, 'html.parser')

    info = soup.find_all('script')
    required_info = re.search(r'const data = \[(.*?)\]', str(info), flags=re.S).group(1)
    #print(required_info)

    str_len = len(required_info)
    #print(f'str_len = {str_len}')

    curr_date = START_DATE
    with open(filename, "w", encoding='utf-8') as output_file:
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

    view_script(URL)
