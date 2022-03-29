from TAQTradesReader import TAQTradesReader
from TAQQuotesReader import TAQQuotesReader
import pandas as pd
import TAQFilter
import os

'''
since we only generate adjusted data in each date folder,
we need to concatenate them into one big dataframe for each stock
'''

'''
The class TAQAdjust can adjust ticker information for specific date and save as pickle file
only tickers in S&P 500 list will be processed
if a stock experienced any kind of stock splitting or dividend paying which may affect adjust factor
we adjust the historical price and size, namely the price and size before such event happens to the level afterwards.
'''


class TAQAdjust(object):
    def __init__(self, date, factors_df, splitting_df):
        self.date = date
        self.factors_df = factors_df
        self.factors_df["Date"] = pd.to_datetime(self.factors_df["Date"])
        self.sp500_reference = TAQFilter.read_tickers(date)
        self.cur_dir = os.getcwd()
        splitting_df["Splitting_date"] = pd.to_datetime(splitting_df["Splitting_date"])
        self.splitting_df = splitting_df

    def adjust_price(self, price, previous_factor, new_factor):
        # historical price divided by previous factor to get unadjusted price
        return price / previous_factor * new_factor

    def adjust_share_volume(self, size, previous_factor, new_factor):
        # left to be validated
        return size * previous_factor / new_factor

    def get_factors(self, ticker_name, date_, splitting_date):
        previous_factor_price = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(date_, format="%Y%m%d"))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Prices"].iloc[0]

        new_factor_price = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(splitting_date))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Prices"].iloc[0]

        previous_factor_volume = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(date_, format="%Y%m%d"))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Shares/Vol"].iloc[0]

        new_factor_volume = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(splitting_date))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Shares/Vol"].iloc[0]

        return previous_factor_price, new_factor_price, previous_factor_volume, new_factor_volume

    def retrieve_trade(self):
        # if you are using trade_test, please change the dir path to trades_test
        sub_path = "data/trades"
        path = os.path.join(self.cur_dir, sub_path)
        whole_path = os.path.join(path, self.date)
        if not os.path.exists(os.path.join(whole_path,"Adjusted")):
            os.mkdir(os.path.join(whole_path,"Adjusted"))
            print("created a sub-dir as a folder to contain all adjusted pickle files")
        tickers_dir = os.listdir(whole_path)
        print("********************************************************************************************")
        print("We are retrieving information from following dir:")
        print(whole_path)
        for ticker in tickers_dir:
            ticker_name = ticker[:-13]
            # filter out useful company tickers according to today's sp500 namelist reference
            if ticker_name in self.sp500_reference:
                NEED_ADJUST = False
                # a necessary filter on stock_splitting factor adjustment
                if ticker_name in list(self.splitting_df["Ticker"]):
                    # if today's date is earlier than stock splitting date
                    # we need to adjust all the historical price in order to match the price gap
                    splitting_date = self.splitting_df.loc[self.splitting_df.Ticker == ticker_name]["Splitting_date"].iloc[0]
                    if pd.to_datetime(self.date, format="%Y%m%d") < pd.to_datetime(splitting_date):
                        NEED_ADJUST = True
                        previous_factor_price, new_factor_price, previous_factor_volume, new_factor_volume = \
                            self.get_factors(ticker_name, self.date, splitting_date)
                dictionary = {}
                print("System is retriving information on {}.".format(ticker_name))
                obj = TAQTradesReader(os.path.join(whole_path, ticker))
                num_rows = obj.getN()
                # print("output of getN: ", num_rows)
                secsFromEpocToMidn = obj.getSecsFromEpocToMidn()
                # print("output of getSecsFromEpocToMidn: ", secsFromEpocToMidn)
                for i in range(num_rows):
                    sub_dic = {}

                    price = obj.getPrice(i)
                    millis = obj.getMillisFromMidn(i)
                    timestamp = obj.getTimestamp(i)
                    size = obj.getSize(i)

                    sub_dic["Seconds_from_Epoc"] = secsFromEpocToMidn
                    sub_dic["Price"] = price
                    sub_dic["Millis"] = millis
                    sub_dic["Timestamp"] = timestamp
                    sub_dic["Size"] = size
                    if NEED_ADJUST:
                        sub_dic["Adjusted_price"] = self.adjust_price(price, previous_factor_price, new_factor_price)
                        sub_dic["Adjusted_size"] = self.adjust_share_volume(size, previous_factor_volume,
                                                                            new_factor_volume)
                    else:
                        sub_dic["Adjusted_price"] = price
                        sub_dic["Adjusted_size"] = size

                    dictionary[i] = sub_dic

                    # use those information to create a dictionary

                    # print("output of getPrice at {}: ".format(i), price)
                    # print("output of getMillisFromMidn at {}: ".format(i), millis)
                    # print("output of getTimestamp at {}: ".format(i), timestamp)
                    # print("output of getSize at {}: ".format(i), size)
                    # print()
                df = pd.DataFrame.from_dict(dictionary, orient="index")
                df["Date"] = pd.to_datetime(self.date, format="%Y%m%d")
                pkl_filename = ticker_name + ".pkl"
                df.to_pickle(os.path.join(os.path.join(whole_path,"Adjusted"),pkl_filename))
        return

    def retrieve_quote(self):
        # if you are using quote_test, please change the dir path to trades_test
        sub_path = "data/quotes"
        path = os.path.join(self.cur_dir, sub_path)
        whole_path = os.path.join(path, self.date)
        if not os.path.exists(os.path.join(whole_path,"Adjusted")):
            os.mkdir(os.path.join(whole_path,"Adjusted"))
            print("created a sub-dir as a folder to contain all adjusted pickle files")
        tickers_dir = os.listdir(whole_path)
        print("********************************************************************************************")
        print("We are retrieving information from following dir:")
        print(whole_path)
        for ticker in tickers_dir:
            ticker_name = ticker[:-13]
            # filter out useful company tickers according to today's sp500 namelist reference
            if ticker_name in self.sp500_reference:
                NEED_ADJUST = False
                # a necessary filter on stock_splitting factor adjustment
                if ticker_name in list(self.splitting_df["Ticker"]):
                    # if today's date is earlier than stock splitting date
                    # we need to adjust all the historical price in order to match the price gap
                    splitting_date = self.splitting_df.loc[self.splitting_df.Ticker == ticker_name]["Splitting_date"].iloc[0]
                    if pd.to_datetime(self.date, format="%Y%m%d") < pd.to_datetime(splitting_date):
                        NEED_ADJUST = True
                        previous_factor_price, new_factor_price, previous_factor_volume, new_factor_volume = \
                            self.get_factors(ticker_name, self.date, splitting_date)

                dictionary = {}
                print("System is retriving information on {}.".format(ticker_name))
                obj = TAQQuotesReader(os.path.join(whole_path, ticker))
                num_rows = obj.getN()
                # print("output of getN: ", num_rows)
                secsFromEpocToMidn = obj.getSecsFromEpocToMidn()
                # print("output of getSecsFromEpocToMidn: ", secsFromEpocToMidn)
                for i in range(num_rows):
                    sub_dic = {}

                    millis = obj.getMillisFromMidn(i)
                    ask_price = obj.getAskPrice(i)
                    bid_price = obj.getBidPrice(i)

                    ask_size = obj.getAskSize(i)
                    bid_size = obj.getBidSize(i)

                    sub_dic["Seconds_from_Epoc"] = secsFromEpocToMidn
                    sub_dic["Millis"] = millis
                    sub_dic["Ask_price"] = ask_price
                    sub_dic["Bid_price"] = bid_price
                    sub_dic["Ask_size"] = ask_size
                    sub_dic["Bid_size"] = bid_size

                    if NEED_ADJUST:
                        sub_dic["Adjusted_bid_price"] = self.adjust_price(bid_price, previous_factor_price,
                                                                          new_factor_price)
                        sub_dic["Adjusted_ask_price"] = self.adjust_price(ask_price, previous_factor_price,
                                                                          new_factor_price)
                        sub_dic["Adjusted_bid_size"] = self.adjust_share_volume(bid_size, previous_factor_volume,
                                                                            new_factor_volume)
                        sub_dic["Adjusted_ask_size"] = self.adjust_share_volume(ask_size, previous_factor_volume,
                                                                                new_factor_volume)
                    else:
                        sub_dic["Adjusted_bid_price"] = bid_price
                        sub_dic["Adjusted_ask_price"] = ask_price
                        sub_dic["Adjusted_bid_size"] = bid_size
                        sub_dic["Adjusted_ask_size"] = ask_size

                    dictionary[i] = sub_dic

                    # use those information to create a dictionary

                    # print("output of getPrice at {}: ".format(i), price)
                    # print("output of getMillisFromMidn at {}: ".format(i), millis)
                    # print("output of getTimestamp at {}: ".format(i), timestamp)
                    # print("output of getSize at {}: ".format(i), size)
                    # print()
                df = pd.DataFrame.from_dict(dictionary, orient="index")
                df["Date"] = pd.to_datetime(self.date, format="%Y%m%d")
                pkl_filename = ticker_name + ".pkl"
                df.to_pickle(os.path.join(os.path.join(whole_path,"Adjusted"), pkl_filename))
        return


if __name__ == "__main__":
    relative_path = "./data"
    trade_path = os.path.join(relative_path, "trades")
    quote_path = os.path.join(relative_path, "quotes")
    trade_dir = os.listdir(trade_path)
    quote_dir = os.listdir(quote_path)
    factor_df = pd.read_csv("S&P500_factors.csv")
    splitting_df = pd.read_csv("splitting_info.csv")

    for trade in trade_dir:
        if not trade.startswith("."):
            date = trade
            TAQAdjust_obj = TAQAdjust(date, factor_df, splitting_df)
            TAQAdjust_obj.retrieve_trade()
    
    for quote in quote_dir:
        if not quote.startswith("."):
            date = quote
            TAQAdjust_obj = TAQAdjust(date, factor_df,splitting_df)
            TAQAdjust_obj.retrieve_quote()

    #concat_date()

