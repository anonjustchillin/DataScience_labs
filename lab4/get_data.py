from bs4 import BeautifulSoup
import csv
import re
import cloudscraper
import os
from deep_translator import GoogleTranslator
import pandas as pd

PROJECT_PATH = 'D:\\uni\\3курс\\Data_Science\\Data_science_labs\\lab4\\data'
SEP = '|'

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }

CSV_FIELDS = ['Id', 'Comment']
FILEPATH = ''


def get_raw_csv(url, name):
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)
    if response.status_code != 200:
        print(f"The request failed with an error {response.status_code}")
        exit()

    soup = BeautifulSoup(response.text, 'lxml')

    reviews = []
    quotes = soup.find_all('div', class_='comment__body-wrapper')
    for x in quotes:
        reviews.append(x.find('p').text)

    filename = name+"_raw.csv"
    global FILEPATH
    filepath = os.path.join(PROJECT_PATH, filename)
    FILEPATH = filepath
    with open(filepath, "w", encoding="utf-8") as output_file:
        csvwriter = csv.writer(output_file, delimiter=SEP)
        csvwriter.writerow(CSV_FIELDS)
        counter = 0
        for review in reviews:
            review = review.replace('\n', ' ')
            csvwriter.writerow([str(counter), review])
            counter += 1
    return

def raw_csv_to_dataframe(name,to_clean=False):
    def clean_data(df, column_name='Comment'):
        def clean_row(r):
            text = r[column_name]

            def rus_to_uk(text):
                translation = GoogleTranslator(source="russian", target="ukrainian").translate(text)
                return translation

            # переклад з рос на українську
            text = rus_to_uk(text)
            # посилання
            text = re.sub(r"https?://\S+|www\.\S+", ' ', text)
            # html теги
            text = re.sub(r"<.*?>", ' ', text)
            # пунктуація
            #text = re.sub(r"[^\w\s]", ' ', text)
            # слова з цифрами
            #text = re.sub(r"\w*\d\w*", ' ', text)
            # цифри
            #text = re.sub(r'\d+', ' ', text)
            text = ' '.join(text.split())
            text = text.lower()

            if len(text) <=3:
                return pd.NA

            return text
        df = df.apply(clean_row, axis=1)
        df.dropna(inplace=True, ignore_index=True)
        return df

    global FILEPATH
    raw_df = pd.read_csv(FILEPATH, sep=SEP, index_col=0)
    print()
    print('RAW DATA')
    print(raw_df.head())
    print(raw_df.tail())
    print()

    filename = name + "_df.csv"
    filepath = os.path.join(PROJECT_PATH, filename)
    raw_df.to_csv(filepath)

    if to_clean:
        clean_df = clean_data(raw_df.copy())
        filename = name + "_df_cleaned.csv"
        filepath = os.path.join(PROJECT_PATH, filename)
        clean_df.to_csv(filepath)

    return

def get_data(url, name):
    get_raw_csv(url, name)
    raw_csv_to_dataframe(name, True)
    return