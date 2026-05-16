import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.models import ARIMA, RandomForestModel, BlockRNNModel
from darts.utils.statistics import check_seasonality, plot_acf, plot_pacf, plot_hist
from darts.metrics import mape, rmse, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings('ignore')

INPUT_DATA_FILE = 'Sales_dataset.xlsx'

def extrapolate(df, col='SALES', predict_to=100, normalize=True):
    def evaluate(data, pred, model_name):
        mape_res = mape(data, pred)
        rmse_res = rmse(data, pred)
        r2_res = r2_score(data, pred)

        print(f"Оцінка моделі: {model_name}")
        print(f"MAPE (Середня абсолютна відсоткова похибка): {mape_res:.2f}%")
        print(f"RMSE (Коренева середньоквадратична похибка): {rmse_res:.2f}")
        print(f"R^2 (Коефіцієнт детермінації): {r2_res:.4f}\n")
        return

    def plot_pred(data, pred, name, days_from_start_pred=None):
        data.plot(label='Original', color='blue')
        pred.plot(label='Prediction', color='red')
        plt.legend()
        plt.grid(alpha=0.4)
        plt.title(f'{name} Forecast {col}')
        if days_from_start_pred is not None:
            start_plot_date = pred.start_time() - pd.Timedelta(days=days_from_start_pred)
            end_plot_date = pred.end_time()
            plt.xlim(start_plot_date, end_plot_date)
        plt.show()

    def arima_method(data):
        train, val = data[:-100], data[-100:]

        model = ARIMA()
        model.fit(train)

        pred_val = model.predict(len(val), series=train)
        plot_pred(data, pred_val, 'ARIMA', 360)
        plot_pred(data, pred_val, 'ARIMA (validation)')

        pred = model.predict(len(val)+predict_to, series=train)
        plot_pred(data, pred, 'ARIMA', 30)
        plot_pred(data, pred, 'ARIMA', 360)
        plot_pred(data, pred, 'ARIMA')

        evaluate(val, pred_val, 'ARIMA')
        return pred

    def random_forest_method(data, period):
        train, val = data[:-100], data[-100:]

        model = RandomForestModel(
            lags=50,
            n_estimators=period,
            max_depth=10,
            random_state=42
        )
        model.fit(series=train)

        pred_val = model.predict(len(val), series=train)
        plot_pred(data, pred_val, 'RANDOM FOREST', 360)
        plot_pred(data, pred_val, 'RANDOM FOREST (validation)')

        pred = model.predict(len(val) + predict_to, series=train)
        plot_pred(data, pred, 'RANDOM FOREST', 30)
        plot_pred(data, pred, 'RANDOM FOREST', 360)
        plot_pred(data, pred, 'RANDOM FOREST')

        evaluate(val, pred_val, 'RANDOM FOREST')
        return pred

    def lstm_method(data, period):
        train, val = data[:-100], data[-100:]

        model = BlockRNNModel(
            model="RNN",
            input_chunk_length=period,
            output_chunk_length=predict_to,
            dropout=0.2,
            n_epochs=50,
            random_state=42
        )
        model.fit(series=train)

        pred_val = model.predict(len(val), series=train)
        plot_pred(data, pred_val, 'LSMT', 360)
        plot_pred(data, pred_val, 'LSMT (validation)')

        pred = model.predict(len(val) + predict_to, series=train)
        plot_pred(data, pred, 'LSMT', 30)
        plot_pred(data, pred, 'LSMT', 360)
        plot_pred(data, pred, 'LSMT')

        evaluate(val, pred_val, 'LSMT')
        return pred

    if normalize:
        df['SALES'] = StandardScaler().fit_transform(df[['SALES']])
        df['TOTAL_PROFIT_LOSS'] = StandardScaler().fit_transform(df[['TOTAL_PROFIT_LOSS']])
        plt.figure(figsize=(16, 5))
        plt.plot(df_short.index, df_short[col])
        plt.title(f'{col} normalized')
        plt.grid(True, alpha=0.4)
        plt.legend('', frameon=False)
        plt.show()

    ts = TimeSeries.from_dataframe(
        df,
        value_cols=col,
        freq='D'
    )
    print()
    is_seasonal, period = check_seasonality(ts)
    print(f'Is {col} seasonal: {is_seasonal}')
    print(f'Periodicity: {period}')
    print()

    plot_acf(ts)
    plt.title(f"ACF {col}")
    plt.show()

    plot_pacf(ts)
    plt.title(f"PACF {col}")
    plt.show()

    plot_hist(ts)
    plt.title(f"Histogram {col}")
    plt.show()

    arima_method(ts)
    random_forest_method(ts, period)
    lstm_method(ts, period)

    return

def show_plots(df, plots, bars):
    def solo_plots():
        for col in df.columns:
            if col in plots:
                plt.figure(figsize=(16, 5))
                df.plot(x='ORDER_DATE', y=col, figsize=(16, 5))
            elif col in bars:
                plt.figure(figsize=(16, 5))
                sns.countplot(x=col, data=df)
                print(df[col].value_counts())
                print()
            else:
                continue
            plt.title(col)
            plt.xticks(rotation=-90)
            plt.grid(True, alpha=0.4)
            plt.legend('', frameon=False)
            plt.show()

    def plot_by_year_month(col):
        df2 = pd.pivot_table(df, columns='MONTH', index='YEAR', aggfunc='sum', values=col)
        df2.sort_index(inplace=True)
        plt.figure(figsize=(16, 5))
        df2.plot(use_index=True, figsize=(16, 5))
        plt.title(f'{col} by Year/Month')
        plt.xticks(rotation=-90)
        plt.grid(True, alpha=0.4)
        plt.show()

        df2 = pd.pivot_table(df, columns='YEAR', index='MONTH', aggfunc='sum', values=col)
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.figure(figsize=(16, 5))
        df2.reindex(month_order).plot(use_index=True)
        plt.title(f'{col} by Month/Year')
        plt.xticks(rotation=-90)
        plt.grid(True, alpha=0.4)
        plt.show()

    def plot_with_2cols(col1, col2):
        plt.figure(figsize=(16, 5))
        plt.plot(df['ORDER_DATE'], df[col1], color='blue', label=col1)
        plt.plot(df['ORDER_DATE'], df[col2], color='orange', label=col2)
        plt.title(f'{col1} and {col2}')
        plt.xticks(rotation=-90)
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.show()

        plt.figure(figsize=(16, 5))
        plt.plot(df['ORDER_DATE'], df[col1], color='blue', label=col1)
        plt.plot(df['ORDER_DATE'], df[col2], color='orange', label=col2)
        plt.title(f'{col1} and {col2}')
        plt.xticks(rotation=-90)
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.show()

        df2 = pd.pivot_table(df, index='MONTH', aggfunc='sum', values=[col1, col2])
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.figure(figsize=(16, 5))
        df2[col1].reindex(month_order).plot(kind='bar', color='blue')
        df2[col2].reindex(month_order).plot(kind='bar', color='orange')
        plt.title(f'{col1} and {col2}')
        plt.xticks(rotation=-90)
        plt.grid(True, alpha=0.4)
        plt.show()

        df2 = pd.pivot_table(df, index='YEAR', aggfunc='sum', values=[col1, col2])
        plt.figure(figsize=(16, 5))
        df2[col1].plot(kind='bar', color='blue')
        df2[col2].plot(kind='bar', color='orange')
        plt.title(f'{col1} and {col2}')
        plt.xticks(rotation=-90)
        plt.grid(True, alpha=0.4)
        plt.show()


    solo_plots()

    plt.figure(figsize=(16, 5))
    sns.countplot(x='MONTH', hue='YEAR', data=df)
    plt.title('Total Count by Month/Year')
    plt.xticks(rotation=-90)
    plt.grid(True, alpha=0.4)
    plt.show()

    plot_by_year_month('SALES')
    plot_by_year_month('TOTAL_PROFIT_LOSS')
    plot_with_2cols('SALES', 'TOTAL_PROFIT_LOSS')
    return

def stats(col):
    mean_col = round(np.mean(col), 4)
    med_col = round(np.median(col), 4)
    var_col = round(np.var(col), 4)
    std_col = round(np.std(col), 4)

    print(f'Length = {len(col)}')
    print(f'Mean = {mean_col}')
    print(f'Median = {med_col}')
    print(f'Variance = {var_col}')
    print(f'Standard deviation = {std_col}')
    return

if __name__ == '__main__':
    print('Opening file...')
    df = pd.read_excel(INPUT_DATA_FILE)

    print(f'Dataset size: {len(df)}')
    print(f'Columns: {df.columns}')

    df.rename(columns={'Estimated Cost Price (50%)\t': 'ESTIMATED_COST_PRICE',
                       'Selling price ': 'SELLING_PRICE',
                        'Profit per unit': 'PROFIT_PER_UNIT',
                        'Total profit / loss' : 'TOTAL_PROFIT_LOSS',
                        'Status ': 'STATUS'}, inplace=True)
    print(f'Columns (renamed): {df.columns}')
    print()
    print('Column types')
    print(df.dtypes)
    print()
    for col in df.columns:
        print(df[col].head())
        print()

    print()
    print(f'{df.isnull().sum().sum()} missing values')
    nan_rows = df[df.isnull().T.any()]
    print(nan_rows)
    df = df.dropna()
    print()

    df.sort_values(by=['ORDER_DATE'], inplace=True)

    cols_plot = ['QUANTITY_ORDERED', 'MSRP',
                 'ESTIMATED_COST_PRICE', 'SELLING_PRICE', 'SALES', 'PROFIT_PER_UNIT',
                 'TOTAL_PROFIT_LOSS']
    cols_bar = ['CUSTOMER_NAME', 'STATUS', 'MONTH', 'YEAR', 'PRODUCT',
                'PRODUCT_CODE', 'CITY', 'COUNTRY', 'DEALSIZE']
    #show_plots(df, cols_plot, cols_bar)
    print()

    start_date = df.at[0, 'ORDER_DATE']
    end_date = df.at[len(df)-1, 'ORDER_DATE']
    print('Missing dates')
    missing_dates = pd.date_range(start=start_date, end=end_date).difference(df['ORDER_DATE'])
    print(f'Count: {len(missing_dates)}')
    print(missing_dates)
    print()

    ###############
    df_short = df.loc[:, ['ORDER_DATE', 'SALES', 'TOTAL_PROFIT_LOSS']].copy()
    print(df_short['ORDER_DATE'].value_counts())
    print()
    print(f'Dataframe length: {len(df_short)}')
    df_short['ORDER_DATE'] = pd.to_datetime(df_short['ORDER_DATE'])
    df_short.set_index('ORDER_DATE', inplace=True)
    df_short = df_short.resample('D').sum()
    print(f'Updated dataframe length: {len(df_short)}')
    df_short['SALES'].interpolate(method='time', inplace=True)
    df_short['TOTAL_PROFIT_LOSS'].interpolate(method='time', inplace=True)
    print(f'{df.isnull().sum().sum()} missing values')
    #print()
    print(df_short.describe())
    stats(df['SALES'])

    plt.figure(figsize=(16, 5))
    plt.plot(df_short.index, df_short['SALES'])
    plt.title('SALES')
    plt.grid(True, alpha=0.4)
    plt.legend('', frameon=False)
    plt.show()

    extrapolate(df_short)

