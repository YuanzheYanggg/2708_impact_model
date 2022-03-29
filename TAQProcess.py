import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed, cpu_count
from TAQMatrices import TAQMatrices

pd.options.mode.chained_assignment = None
'''
since we already save an all-period trade record and an all-period quote record for each ticker as pickled file
our TAQProcess class will help us clean the corresponding trade and quote and computing data matrix 
that will be further used in building impact model.

we may need further discuss on how to deal with other outliers, here I just replace them by nan
but a discussion on do we have to fill those nan values before calculated statistics is still needed
'''
# data storage

arrival_price_matrix = {}
terminal_price_matrix = {}
ret_std_matrix = {}
vwap_330_matrix = {}
vwap_400_matrix = {}
imbalance_share_matrix = {}
daily_volume_sum_matrix = {}


class TAQProcess(object):
    def __init__(self):
        sp_df = pd.read_csv("./S&P500_factors.csv")
        self.ticker_name_list = list(sp_df.Ticker.unique())
        self.rolling_window = 5
        self.threshold_error = 5 * 1e-5

    def clean_trades(self, df):
        daily_mean = df.groupby("Date").mean()
        rolling_mean = daily_mean.rolling(self.rolling_window).mean()
        rolling_std = daily_mean.rolling(self.rolling_window).std()

        def cleaning_trade_outlier(x):
            date = x.Date
            mean = rolling_mean.loc[rolling_mean.index == date]
            std = rolling_std.loc[rolling_std.index == date]
            # for first k days, since we do not have enough historical rolling window size
            # we decide to skip those days and leave the data unchanged
            if mean.isnull().sum().sum() != len(mean.columns):
                # replace price,size,adj_price,adj_size of outliers with np.nan
                mean_price = mean.Adjusted_price.iloc[0]
                std_price = std.Adjusted_price.iloc[0]
                if (x.Adjusted_price > mean_price + 2 * std_price + self.threshold_error * mean_price) or \
                        (x.Adjusted_price < mean_price - 2 * std_price + self.threshold_error * mean_price):
                    x.Price = np.nan
                    x.Adjusted_price = np.nan
                    x.Size = np.nan
                    x.Adjusted_size = np.nan
            return x

        df = df.apply(lambda x: cleaning_trade_outlier(x), axis=1)
        return df

    def clean_quotes(self, df):
        daily_mean = df.groupby("Date").mean()
        rolling_mean = daily_mean.rolling(self.rolling_window).mean()
        rolling_std = daily_mean.rolling(self.rolling_window).std()

        def cleaning_quote_outlier(x):
            date = x.Date
            mean = rolling_mean.loc[rolling_mean.index == date]
            std = rolling_std.loc[rolling_std.index == date]
            # for first k days, since we do not have enough historical rolling window size
            # we decide to skip those days and leave the data unchanged
            if mean.isnull().sum().sum() != len(mean.columns):
                # replace price,size,adj_price,adj_size of outliers with np.nan
                mean_ask_price = mean.Adjusted_ask_price.iloc[0]
                mean_bid_price = mean.Adjusted_bid_price.iloc[0]
                std_ask_price = std.Adjusted_ask_price.iloc[0]
                std_bid_price = std.Adjusted_bid_price.iloc[0]
                if (x.Adjusted_ask_price > mean_ask_price + 2 * std_ask_price + self.threshold_error * mean_ask_price) or \
                        (x.Adjusted_ask_price < mean_ask_price - 2 * std_ask_price + self.threshold_error * mean_ask_price):
                    x.Ask_price = np.nan
                    x.Adjusted_ask_price = np.nan
                    x.Ask_size = np.nan
                    x.Adjusted_ask_size = np.nan

                if (x.Adjusted_bid_price > mean_bid_price + 2 * std_bid_price + self.threshold_error * mean_bid_price) or \
                        (x.Adjusted_bid_price < mean_bid_price - 2 * std_bid_price + self.threshold_error * mean_bid_price):
                    x.Bid_price = np.nan
                    x.Adjusted_bid_price = np.nan
                    x.Bid_size = np.nan
                    x.Adjusted_bid_size = np.nan
            return x

        df = df.apply(lambda x: cleaning_quote_outlier(x), axis=1)
        return df

    def process_data(self, num_cpus):

        print("Parallel process trade data:")
        trades_result = Parallel(n_jobs=num_cpus, backend='loky')(
            delayed(self.traverse_ticker_trades)(ticker) for ticker in self.ticker_name_list)

        for res in trades_result:
            if not res[0][1] or not res[1][1] or not res[2][1] or not res[3][1]:
                continue
            vwap_330_matrix[res[0][0]] = res[0][1]
            vwap_400_matrix[res[1][0]] = res[1][1]
            imbalance_share_matrix[res[2][0]] = res[2][1]
            daily_volume_sum_matrix[res[3][0]] = res[3][1]

        print("Parallel process quote data:")
        quotes_result = Parallel(n_jobs=num_cpus, backend='loky')(
            delayed(self.traverse_ticker_quotes)(ticker) for ticker in self.ticker_name_list)

        for res in quotes_result:
            if not res[0][1] or not res[1][1] or not res[2][1]:
                continue
            arrival_price_matrix[res[0][0]] = res[0][1]
            terminal_price_matrix[res[1][0]] = res[1][1]
            ret_std_matrix[res[2][0]] = res[2][1]

        return


    # for ticker in sp500 list
    def traverse_ticker_trades(self,ticker):
        print("processing {} trade data".format(ticker))
        # if you are using trade_test, please change the dir path to trades_test
        trade_dir = os.path.join(os.getcwd(), "data/trades")
        trade_dates_list = os.listdir(trade_dir)
        trades_columns = ["Seconds_from_Epoc", "Price", "Millis", "Timestamp", "Size", "Adjusted_price",
                          "Adjusted_size", "Date"]
        df = pd.DataFrame(columns=trades_columns)
        ticker_filename = ticker + ".pkl"
        for trade_date in trade_dates_list:
            if trade_date.startswith("."):
                continue
            sub_path = os.path.join(os.path.join(trade_dir, trade_date), "Adjusted")
            # if the ticker is not in sp500 namelist that date, we skip to next date
            if ticker_filename not in os.listdir(sub_path):
                continue
            data = pd.read_pickle(os.path.join(sub_path, ticker_filename))
            df = pd.concat([df, data], ignore_index=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Date", "Millis"])
        df["Datetime"] = df["Seconds_from_Epoc"] + df["Millis"] / 1000
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
        df.reset_index(drop=True, inplace=True)

        if df.shape[0] == 0:
            return (ticker, None), (ticker, None), (ticker, None), (ticker, None)

        df = self.clean_trades(df)

        # generate our data matrix for this specific ticker

        TAQMatrices_trade_obj = TAQMatrices(df, "trade")
        TAQMatrices_trade_obj.generate_matrices()
        vwap_330_dic = TAQMatrices_trade_obj.vwap_330_dic
        vwap_400_dic = TAQMatrices_trade_obj.vwap_400_dic
        imbalance_share_dic = TAQMatrices_trade_obj.imbalance_share_dic
        daily_volume_sum_dic = TAQMatrices_trade_obj.daily_volume_sum_dic

        print("done processing {} trade data".format(ticker))
        return (ticker, vwap_330_dic), (ticker, vwap_400_dic), (ticker, imbalance_share_dic), \
               (ticker, daily_volume_sum_dic)

    def traverse_ticker_quotes(self, ticker):
        print("processing {} quote data".format(ticker))
        quote_dir = os.path.join(os.getcwd(), "data/quotes")
        quote_dates_list = os.listdir(quote_dir)

        quotes_columns = ["Seconds_from_Epoc", "Millis", "Ask_price", "Bid_price", "Ask_size", "Bid_size",
                          "Adjusted_bid_price",
                          "Adjusted_ask_price", "Adjusted_bid_size", "Adjusted_ask_size", "Date"]

        df = pd.DataFrame(columns=quotes_columns)
        ticker_filename = ticker + ".pkl"
        for quote_date in quote_dates_list:
            if quote_date.startswith("."):
                continue
            sub_path = os.path.join(os.path.join(quote_dir, quote_date), "Adjusted")
            # if the ticker is not in sp500 namelist that date, we skip to next date
            if ticker_filename not in os.listdir(sub_path):
                continue
            data = pd.read_pickle(os.path.join(sub_path, ticker_filename))
            df = pd.concat([df, data], ignore_index=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Date", "Millis"])
        df["Datetime"] = df["Seconds_from_Epoc"] + df["Millis"] / 1000
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
        df.reset_index(drop=True, inplace=True)

        if df.shape[0] == 0:
            return (ticker, None), (ticker, None), (ticker, None)
        df = self.clean_quotes(df)

        TAQMatrices_quote_obj = TAQMatrices(df, "quote")
        TAQMatrices_quote_obj.generate_matrices()
        arrival_price_dic = TAQMatrices_quote_obj.arrival_price_dic
        #print(arrival_price_dic)
        terminal_price_dic = TAQMatrices_quote_obj.terminal_price_dic
        #print(terminal_price_dic)
        ret_std_dic = TAQMatrices_quote_obj.ret_std_dic
        #print(ret_std_dic)

        print("done processing {} quote data".format(ticker))
        return (ticker, arrival_price_dic),(ticker, terminal_price_dic),(ticker, ret_std_dic)

if __name__ == "__main__":
    TAQProcess_obj = TAQProcess()
    TAQProcess_obj.process_data(-1)

    arrival_price_df = pd.DataFrame(arrival_price_matrix)
    arrival_price_df.to_csv("arrival_price.csv")
    terminal_price_df = pd.DataFrame(terminal_price_matrix)
    terminal_price_df.to_csv("terminal_price.csv")
    ret_std_df = pd.DataFrame(ret_std_matrix)
    ret_std_df.to_csv("ret_std.csv")
    vwap_330_df = pd.DataFrame(vwap_330_matrix)
    vwap_330_df.to_csv("vwap_330.csv")
    vwap_400_df = pd.DataFrame(vwap_400_matrix)
    vwap_400_df.to_csv("vwap_400.csv")
    imbalance_share_df = pd.DataFrame(imbalance_share_matrix)
    imbalance_share_df.to_csv("imbalance_share.csv")
    daily_volume_sum_df = pd.DataFrame(daily_volume_sum_matrix)
    daily_volume_sum_df.to_csv("daily_volume_sum.csv")
