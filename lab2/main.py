import pandas as pd
import matplotlib.pyplot as plt
import math as mt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import re
from bs4 import BeautifulSoup as bs
import requests
import os.path
from statsmodels.tsa.stattools import adfuller

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
def analyze_LSM(filename, interval = 0.5):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    orig_len = len(df)

    def plot_reg(x, y_real, y_pred, title, predictions=False):
        plt.figure(figsize=(15, 6))
        if predictions:
            plt.vlines(x=[orig_len], ymin=0, ymax=y_pred[-1], color='red', linestyle='--')
            plt.plot(x[0:orig_len], y_real, color='blue')
            plt.plot(x[orig_len:], y_pred, color='red')
        else:
            plt.scatter(x, y_real, color='blue')
            plt.plot(x, y_pred, color='red')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.grid(True)

        plt.show()


    def view_score(y_real, y_pred, title):
        print(title)
        mae = mean_absolute_error(y_real, y_pred)
        rmse = root_mean_squared_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')
        print(f'R2: {r2}')


    #x_data = df.index.to_list()
    x_data = np.arange(0, len(df), 1)
    y_data = df[COL_NAME].to_list()

    #print(len(x_data))
    #print(len(y_data))

    ########################### split data (70/30)
    split_idx = int(len(df) - (len(df)*30/100))
    #print(split_idx)
    train_X = np.array(x_data[0:split_idx]).reshape(-1, 1)
    train_Y = np.array(y_data[0:split_idx]).reshape(-1, 1)
    test_x = np.array(x_data[split_idx:]).reshape(-1, 1)
    test_y = np.array(y_data[split_idx:]).reshape(-1, 1)

    ####################### lin reg
    model_lin = LinearRegression()
    model_lin.fit(train_X, train_Y)

    Y_pred_lin = model_lin.predict(train_X)
    y_pred_lin = model_lin.predict(test_x)

    view_score(test_y, y_pred_lin, 'Linear Regression')

    plot_reg(x_data, y_data, np.concatenate((Y_pred_lin,y_pred_lin)), 'Linear Regression')
    plot_reg(test_x, test_y, y_pred_lin, 'Linear Regression')

    print(PRINT_SEP)

    ####################### poly LSM
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    train_X_lsm = poly.fit_transform(train_X)
    test_x_lsm = poly.transform(test_x)
    w = np.linalg.lstsq(train_X_lsm, train_Y, rcond=None)
    #print(w)

    Y_pred_lsm = np.dot(train_X_lsm, w[0])
    y_pred_lsm = np.dot(test_x_lsm, w[0])

    #print(w[0])
    #print(y_pred_lsm)

    view_score(test_y, y_pred_lsm, 'LSM Polynomial (order=2) Regression')

    plot_reg(x_data, y_data, np.concatenate((Y_pred_lsm, y_pred_lsm)), 'LSM Polynomial (order=2) Regression')
    plot_reg(test_x, test_y, y_pred_lsm, 'LSM Polynomial (order=2) Regression')

    print(PRINT_SEP)

    ############ FORECAST :D
    half_idx = int(len(df)*interval)
    new_x_data = np.arange(len(df), len(df)+half_idx+1, 1)
    prep_new_x_data = np.array(new_x_data).reshape(-1, 1)

    whole_x_data = np.concatenate((x_data, new_x_data))

    # lin reg
    new_y_pred_lin = model_lin.predict(prep_new_x_data)
    list_new_y_pred_lin = list(new_y_pred_lin.ravel())

    plot_reg(whole_x_data, y_data, list_new_y_pred_lin, 'Linear Regression Forecasting', True)

    # lsm poly
    new_x_lsm = poly.transform(prep_new_x_data)
    new_y_pred_lsm = np.dot(new_x_lsm, w[0])
    list_new_y_pred_lsm = list(new_y_pred_lsm.ravel())

    plot_reg(whole_x_data, y_data, list_new_y_pred_lsm, 'LSM Polynomial (order=2) Forecasting', True)

    return


# smoothing
class Sample:
    def __init__(self, x, t, v):
        self.location = x
        self.velocity = v
        self.time = t

    def __repr__(self):
        return f"Sample({self.location}, {self.velocity}, {self.time})"


class AlphaBetaGammaFilter:
    def __init__(self, init_sample, alpha=1.0, beta=0.1, gamma=0.0, velocity=1.0, acceleration=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.velocity_list = [velocity]
        self.acceleration_list = [acceleration]
        self.sample_list = [init_sample]
        self.locations = [init_sample.location]
        self.errors = []
        self.predictions = []

    @property
    def last_sample(self):
        return self.sample_list[-1]

    @property
    def last_velocity(self):
        return self.velocity_list[-1]

    @property
    def last_acceleration(self):
        return self.acceleration_list[-1]

    def add_sample(self, s: Sample):
        delta_t = s.time - self.last_sample.time
        expected_location, expected_velocity = self.predict(delta_t)

        error = s.location - expected_location
        location = expected_location + self.alpha * error
        v = expected_velocity + (self.beta / delta_t) * error
        a = self.last_acceleration + (self.gamma * (error / delta_t**2))

        # for debugging and results
        self.velocity_list.append(v)
        self.acceleration_list.append(a)
        self.locations.append(location)
        self.sample_list.append(s)
        self.errors.append(error)

    def predict(self, t):
        # x+(t*v)
        prediction = self.last_sample.location + (t * self.last_velocity)
        # v+(t*a)
        prediction_v = self.last_velocity + (t * self.last_acceleration)

        # for debugging and results
        self.predictions.append(prediction)
        return prediction, prediction_v


def smoothing(filename, alpha, beta, vel, gamma=0, acc=0):
    df = pd.read_csv(filename, encoding='utf-8', index_col=0)
    x_data = np.arange(0, len(df), 1)
    y_data = df[COL_NAME].to_list()

    def plot_smooth(x, y_real, y_pred, title):
        plt.figure(figsize=(15, 6))
        plt.plot(x, y_real, color='blue', alpha=0.5, label='actual data')
        plt.plot(x, y_pred, color='red', label='smoothed data')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)

        plt.show()

    samples = []
    for i in range(len(df)):
        x = x_data[i]
        y = y_data[i]
        samples.append(Sample(y, x, 0))

    filter = AlphaBetaGammaFilter(samples[0],
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  velocity=vel,
                                  acceleration=acc) # results change with beta
    for sample in samples[1:]:
        filter.add_sample(sample)

    y_pred = filter.predictions
    y_pred.insert(0, y_data[0])

    print('Alpha-Beta-Gamma Filter')
    print(f'alpha={alpha}, beta={beta}, gamma={gamma}')
    print(f'velocity={vel}, acceleration={acc}')
    print()
    comp_df = pd.DataFrame({"y_real": y_data, "y_pred": y_pred}, index=x_data)
    print(comp_df.head())
    print(comp_df.tail())
    plot_smooth(x_data, y_data, y_pred, 'Alpha-Beta-Gamma Filter')

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

    # check_stationarity(CLEANED_PATH, True)
    # fill_empty_cells(CLEANED_PATH, output_filename=CLEANED_NONA_PATH)
    # output_stats_graph(filename=CLEANED_PATH, comment=f'Cleaned {COL_NAME} (with NA)')
    # print(PRINT_SEP)
    # check_stationarity(CLEANED_PATH, False)
    # output_stats_graph(filename=CLEANED_NONA_PATH, comment=f'Cleaned {COL_NAME} (without NA)')
    # print(PRINT_SEP)
    # analyze_LSM(CLEANED_NONA_PATH)
    # print(PRINT_SEP)
    #smoothing(CLEANED_NONA_PATH, 0.5, 0.1, 1.0)
