import unittest
from TAQMatrices import TAQMatrices
import pandas as pd


class Test_TAQAdjust(unittest.TestCase):

    def test1(self):
        # we create sub-folders trades_test and quotes_test as our test case
        # it is time-saving especially under the circumstance of huge dataset
        # we comment out this part of code because we already compiled and run it on our local device
        # and the result is already saved to local files

        # this Unit test is just to show everything is running perfectly fine.

        # we fake some dataframe representing our dataframe after cleaning
        # we just want to make sure that the computation in TAQMatrix is correct

        # test trade
        trades_data = {"Seconds_from_Epoc" :[1,2,3,4,5,6,7,8,9,10],
                          "Price" : [10.1,10.2,10.3,10.4,10.5,11,11.1,11.2,11.3,11.4],
                          "Millis" : [1,2,3,4,5,6,7,8,9,10],
                          "Timestamp" : [1,2,3,4,5,6,7,8,9,10],
                          "Size" : [100,200,300,400,500,500,400,300,200,100],
                          "Adjusted_price": [10.1,10.2,10.3,10.4,10.5,11,11.1,11.2,11.3,11.1],
                          "Adjusted_size": [100,200,300,400,500,500,400,300,200,100],
                          "Date" : [pd.to_datetime("20070907") for i in range(10)],
                          "Datetime" : [pd.to_datetime("2007-09-07 13:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 14:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 15:00", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 15:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 16:00", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 16:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 17:00", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 17:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 19:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 20:00", format="%Y-%m-%d %H:%M")]}

        trade_df = pd.DataFrame(trades_data)


        TAQMatrices_trade_obj = TAQMatrices(trade_df,"trade")
        TAQMatrices_trade_obj.generate_matrices()
        vwap_330_dic = TAQMatrices_trade_obj.vwap_330_dic
        vwap_400_dic = TAQMatrices_trade_obj.vwap_400_dic
        imbalance_share_dic = TAQMatrices_trade_obj.imbalance_share_dic
        daily_volume_sum_dic = TAQMatrices_trade_obj.daily_volume_sum_dic


        self.assertAlmostEquals(vwap_330_dic[pd.to_datetime("20070907")],10.73 , 2)
        self.assertAlmostEquals(vwap_400_dic[pd.to_datetime("20070907")],10.74 , 2)
        self.assertAlmostEquals(imbalance_share_dic[pd.to_datetime("20070907")],2700 , 3)
        self.assertAlmostEquals(daily_volume_sum_dic[pd.to_datetime("20070907")], 3000, 3)



        # test quote
        quotes_data = {"Adjusted_bid_price": [10.1,10.2,10.3,10.4,10.5,11,11.1,11.2,11.3,11.1],
                        "Adjusted_ask_price": [10.11,10.21,10.32,10.41,10.52,11.01,11.11,11.2,11.32,11.11],
                          "Adjusted_bid_size": [100,200,300,400,500,500,400,300,200,100],
                          "Adjusted_ask_size": [100,200,300,400,500,500,400,300,200,100],
                          "Date" : [pd.to_datetime("20070907") for i in range(10)],
                          "Datetime" : [pd.to_datetime("2007-09-07 13:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 14:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 15:00", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 15:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 16:00", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 16:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 17:00", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 17:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 19:30", format="%Y-%m-%d %H:%M"),
                                        pd.to_datetime("2007-09-07 20:00", format="%Y-%m-%d %H:%M")]}

        quote_df = pd.DataFrame(quotes_data)


        TAQMatrices_quote_obj = TAQMatrices(quote_df, "quote")
        TAQMatrices_quote_obj.generate_matrices()
        arrival_price_dic = TAQMatrices_quote_obj.arrival_price_dic
        terminal_price_dic = TAQMatrices_quote_obj.terminal_price_dic
        ret_std_dic = TAQMatrices_quote_obj.ret_std_dic

        self.assertAlmostEquals(arrival_price_dic[pd.to_datetime("20070907")],10.307 , 3)
        self.assertAlmostEquals(terminal_price_dic[pd.to_datetime("20070907")],11.145 , 3)
        self.assertAlmostEquals(ret_std_dic[pd.to_datetime("20070907")],0 , 3)

if __name__ == "__main__":
    unittest.main()