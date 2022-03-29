import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


class TAQMatrices(object):
    def __init__(self, df, type):
        self.df = df
        self.date_list = list(pd.to_datetime(self.df.Date.unique()))
        self.type = type
        self.arrival_price_dic = {}
        self.terminal_price_dic = {}
        self.ret_std_dic = {}
        self.vwap_330_dic = {}
        self.vwap_400_dic = {}
        self.imbalance_share_dic = {}
        self.daily_volume_sum_dic = {}

    def generate_matrices(self):

        # Compute 2-minute mid-quote returns
        # Compute total daily volume
        # Compute arrival price – Average of first five mid-quote prices
        # Compute imbalance between 9:30 and 3:30
        # Compute volume-weighted average price between 9:30 and 3:30
        # Compute volume-weighted average price between 9:30 and 4:00 (This will
        # be used to compute average daily value of imbalance as described later in
        # this document.)
        # Compute terminal price at 4:00 – Average of last five mid-quote prices
        for date in self.date_list:
            date_ = date.strftime(format="%Y%m%d")
            if self.type == "trade":
                # since we used UTC time-zone, we should be consistent with ourselves here.
                self.vwap_330_dic[date] = self.get_vwap(date_, "13:30", "19:30")
                self.vwap_400_dic[date] = self.get_vwap(date_, "13:30", "20:00")
                self.daily_volume_sum_dic[date] = self.get_total_daily_volume(date_)
                self.imbalance_share_dic[date] = self.get_imbalance_share(date_, 0.00001)
            elif self.type == "quote":
                self.arrival_price_dic[date] = self.get_arrival_price(date_)
                self.terminal_price_dic[date] = self.get_terminal_price(date_)
                self.ret_std_dic[date] = self.get_mid_quote_returns_std(date_)
        return

    ##############################################################
    # under type == quote, because we want to use mid-quote data #
    ##############################################################

    # if all the top 5 quotes are np.nan, it will return np.nan aswell

    def get_arrival_price(self, date):
        # comment out the choice you don't like

        # choice 1:
        # we can ignore the Nan price by just output Nan
        sub_df = self.df.loc[self.df.Date == pd.to_datetime(date)][:5]

        # choice 2:
        # since some stock in specific date do not have prices
        # we decided to use open price as arrival price
        # sub_df = self.df.loc[self.df.Date == pd.to_datetime(date)]
        # sub_df[["Adjusted_bid_price", "Adjusted_ask_price"]] = sub_df[
        #    ["Adjusted_bid_price", "Adjusted_ask_price"]].fillna(method="bfill")
        # sub_df = sub_df[:5]

        arrival_prices = (sub_df["Adjusted_bid_price"] + sub_df["Adjusted_ask_price"]) / 2
        print(arrival_prices)
        arrival_price = arrival_prices.mean()
        #print("arrival_price: {}".format(arrival_price))
        return arrival_price

    def get_terminal_price(self, date):
        # comment out the choice you don't like

        # choice 1:
        # we can ignore the Nan price by just output Nan
        sub_df = self.df.loc[self.df.Date == pd.to_datetime(date)][-5:]

        # choice 2:
        # since some stock in specific date do not have prices
        # we decided to use close price as terminal price
        # sub_df = self.df.loc[self.df.Date == pd.to_datetime(date)]
        # sub_df[["Adjusted_bid_price", "Adjusted_ask_price"]] = sub_df[
        #    ["Adjusted_bid_price", "Adjusted_ask_price"]].fillna(method="ffill")
        # sub_df = sub_df[-5:]

        terminal_prices = (sub_df["Adjusted_bid_price"] + sub_df["Adjusted_ask_price"]) / 2
        terminal_price = terminal_prices.mean()
        #print("terminal_price: {}".format(terminal_price))
        return terminal_price

    def get_mid_quote_returns_std(self, date):
        sub_df = self.df.loc[self.df.Date == pd.to_datetime(date)]
        sub_df = sub_df[["Datetime", "Adjusted_bid_price", "Adjusted_ask_price"]]
        sub_df["Adjusted_mid_quote"] = (sub_df["Adjusted_bid_price"] + sub_df["Adjusted_ask_price"]) / 2
        freq = "2t"
        r = sub_df.resample(freq, closed="left", label="right", on="Datetime")
        first = r.agg("first")
        last = r.agg("last")
        ret_df = last[["Adjusted_mid_quote"]] / first[["Adjusted_mid_quote"]] - 1
        ret_df.rename(columns={"Adjusted_mid_quote": freq + "_ret"}, inplace=True)
        return ret_df[freq + "_ret"].std() * np.sqrt(6.5 * 60 * 60 / 2)

    ###############################################################
    # under type == trade, because we only need to use trade data #
    ###############################################################

    # Eg\\ date: 20070620   start_time: 9:30

    def get_vwap(self, date, start_time, end_time):
        start_datetime = "{} {}".format(date, start_time)
        start_datetime = pd.to_datetime(start_datetime, format="%Y%m%d %H:%M")
        end_datetime = "{} {}".format(date, end_time)
        end_datetime = pd.to_datetime(end_datetime, format="%Y%m%d %H:%M")

        sub_df = self.df.loc[(self.df.Datetime >= start_datetime) & (self.df.Datetime <= end_datetime)]
        vwap = ((sub_df["Adjusted_price"] * sub_df["Adjusted_size"]).sum()) / sub_df["Adjusted_size"].sum()
        #print(end_time + " vwap is: {}".format(vwap))
        return vwap

    def get_imbalance_share(self, date, tolerance):
        sub_df = self.df.loc[self.df.Date == pd.to_datetime(date)]
        prev = sub_df["Adjusted_price"][:-1]
        cur = sub_df["Adjusted_price"][1:]
        buy_init = (cur.reset_index() > prev.reset_index() + tolerance)[["Adjusted_price"]]
        buy_init.rename(columns={"Adjusted_price": "buy_init"}, inplace=True)
        sell_init = (cur.reset_index() < prev.reset_index() - tolerance)[["Adjusted_price"]]
        sell_init.rename(columns={"Adjusted_price": "sell_init"}, inplace=True)

        # we do not care about first trade,
        # since we can only infer the trade init side by compare the second trade with that first one
        sub_df = sub_df.iloc[1:].reset_index(drop=True)
        sub_df = pd.concat([sub_df, pd.DataFrame(buy_init)], axis=1)
        sub_df = pd.concat([sub_df, pd.DataFrame(sell_init)], axis=1)
        total_imbalances = []

        def get_imbalance(x, lst):
            if x.buy_init:
                lst.append(x.Adjusted_size)
            if x.sell_init:
                lst.append(-x.Adjusted_size)
            return

        sub_df.apply(lambda x: get_imbalance(x, total_imbalances), axis=1)
        total_imbalance = np.sum(total_imbalances)
        #print("total_imbalance is : {}".format(total_imbalance))
        return total_imbalance

    def get_total_daily_volume(self, date):
        sub_df = self.df.loc[self.df.Date == pd.to_datetime(date)]
        ans = sub_df["Adjusted_size"].sum()
        #print("daily_volume is : {}".format(ans))
        return ans
