# 2708_impact_model

In this prompt, we will build an impact model using public data from 2007-06-20 to 2007-09-20.

The functionalities include: 
1)	TAQTradeReader: import the trade data 
2)	TAQQuotesReader: import the quote data
3)	TAQAdjust: filter out only S&P 500 listed tickers and adjust the stock price and size for stock splitting/ stock buyback.
4)	TAQProcess: process ticker by ticker with multi-threading to save the stocks feature values in the matrix form. 
5)	TAQRegression: perform non-linear regression to compute market impact parameters and perform residual analysis for the regression model
6)	TAQFilter: extract S&P 500 stocks list and stock splitting information. 
7)	TAQMatrices: perform tools for matrix construction in the TAQProcess steps.

The working flow in this project will be from TAQTradeReader - TAQFilter - TAQ_Adjust - TAQ_Process - TAQ_Regression

By noticing that the tickers data are independent with each other, we implement a parallel processing mechanism using Joblib which enables us to run multiple tasks at the same time. The maximum tasks are confined by how many CPU you have with your device.

Since Cleaning and Computing matrix is very time consuming, we only show the regression output in the window between 20070907 and 20070920. If you are interested in exploring more data in different date scope, feel free to check it out.
