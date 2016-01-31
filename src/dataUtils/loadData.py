import pandas as pd
import numpy as np

from datetime import datetime
import os
import pdb

DATAROOT = "/home/ryan/ml/harrisTrader/data/"

def close_prices(symbol):
    filename = DATAROOT + symbol + ".csv"
    history = pd.read_csv(filename, index_col=0)
    return history.values[:,5].size

def percent_change_single(symbol):
    filename = DATAROOT + symbol + ".csv"
    history = pd.read_csv(filename, index_col=0)
    np_data = history.values
    return (np_data[:,5] - np_data[:,1]) / np_data[:,1]

def percent_changes(symbols):
    return np.array([percent_change_single(s) for s in symbols])
# with more data can't return everything in one matrix

def get_list_dates(symbol):
    filename = DATAROOT + symbol + ".csv"
    try:
        history = pd.read_csv(filename)
        np_data = history.values
        start = np_data[-1,0]
        end = np_data[0,0]
        return (start, end)
    except:
        print "unexpected error on reading data for symbol: ", symbol
    return ("1000-01-01", "1000-01-01")

def get_all_symbols():
    return [x[:-4] for x in os.listdir(DATAROOT)]

def symbols_in_int(d1, d2):
    symbols = get_all_symbols()
    # naively
    int_symbols = []
    for symbol in symbols:
        dates = get_list_dates(symbol)
        if stock_date_from_str(dates[0]) <= d1 and stock_date_from_str(dates[1]) >= d2:
            int_symbols.append(symbol)
    return int_symbols

def stock_date_from_str(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

def main():
    start = stock_date_from_str("1995-01-01")
    end = stock_date_from_str("2005-01-01")
    symbols = symbols_in_int(start, end)
    data = percent_changes(symbols)
    print "stocks in range", len(symbols)
    pdb.set_trace()

if __name__ == "__main__": main()
