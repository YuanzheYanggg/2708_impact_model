import pandas as pd
import numpy as np

def collect_ticker_ret():
    df = pd.read_excel("./data/s&p500.xlsx", sheet_name="WRDS", engine="openpyxl")
    df = df[["Names Date", "Trading Symbol", "Returns", "Volume", "Price or Bid/Ask Average", "Shares Outstanding"]]
    df.dropna(inplace=True)
    df.rename(columns={"Names Date": "Date", "Trading Symbol": "Ticker"}, inplace=True)
    df["Turnover"] = df["Volume"]/(df["Price or Bid/Ask Average"] * df["Shares Outstanding"])
    df["Date"] = df.Date.apply(lambda x: str(x)[:-2])
    df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')

    df.to_csv("S&P500_rets.csv", index=False)
    return


def extract_tickers():
    df = pd.read_excel("./data/s&p500.xlsx", sheet_name="WRDS", engine="openpyxl")
    # since same ticker will possibly have different adjust factors for different dates
    # so we need to include both dates and ticker symbol as our search key

    new_df = df.groupby(["Names Date","Trading Symbol","Cumulative Factor to Adjust Prices",
                         "Cumulative Factor to Adjust Shares/Vol"]).size().reset_index(name="Freq")

    # rename dates and ticker names
    new_df.rename(columns={"Names Date": "Date", "Trading Symbol": "Ticker"}, inplace=True)
    # reformat the Date column, more recognizable and easier to use
    new_df["Date"] = new_df.Date.apply(lambda x: str(x)[:-2])
    new_df["Date"] = pd.to_datetime(new_df["Date"], format='%Y%m%d')

    # dump the adjust factors into local csv file, so we can directly use it next time
    new_df.to_csv("S&P500_factors.csv",index=False)

    # if we look at the group by dataframe without a column of time date, we can extract information
    # about which companies/ tickers are experiencing stock split under specific period
    new_df_2 = df.groupby(["Trading Symbol", "Cumulative Factor to Adjust Prices",
                           "Cumulative Factor to Adjust Shares/Vol"]).size().reset_index(name="Freq")
    new_df_2 = new_df_2.groupby(["Trading Symbol"]).count()
    stock_spliting_ticker_list = list(new_df_2.loc[new_df_2.Freq > 1].index)

    splitting_dates = []
    for stock in stock_spliting_ticker_list:
        date = find_splitting_date(new_df, stock)
        splitting_dates.append(date)
    temp_dic = {"Ticker": stock_spliting_ticker_list, "Splitting_date": splitting_dates}
    splitting_df = pd.DataFrame(temp_dic)
    splitting_df.to_csv("splitting_info.csv", index=False)
    return


def find_splitting_date(df,ticker):
    sub_df = df.loc[df.Ticker == ticker]
    fp = "Cumulative Factor to Adjust Prices" # define abbreviation for Cumulative Factor to Adjust Prices
    fs = "Cumulative Factor to Adjust Shares/Vol"  # define abbreviation for Cumulative Factor to Adjust Shares/Vol
    date = None
    #
    for i in range(1,sub_df.shape[0]):
        # if either of fp or fs changes from previous date, we discover a splitting date
        # Here, according to our analysis, all the stocks should have at most one stock splitting during period
        # So we can make it easy and only record the first stock splitting date
        if sub_df.iloc[i][fp] != sub_df.iloc[i-1][fp] or sub_df.iloc[i][fs] != sub_df.iloc[i-1][fs]:
            date = sub_df.iloc[i]["Date"]
            break
    return date


def read_tickers(date_string):
    date = pd.to_datetime(date_string, format="%Y%m%d")
    df = pd.read_csv("./S&P500_factors.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    lst = list(df.loc[df.Date == date]["Ticker"])
    return lst


if __name__ == "__main__":
    #collect_ticker_ret()
    extract_tickers()
