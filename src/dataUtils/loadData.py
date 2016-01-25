import pandas as pd
import numpy as np
import os

DATAROOT = "../../data/"

def close_prices(symbol):
    filename = DATAROOT + symbol + ".csv"
    history = pd.read_csv(filename, index_col=0)
    return history.values[:,5].size

def percent_changes(symbol):
    filename = DATAROOT + symbol + ".csv"
    history = pd.read_csv(filename, index_col=0)
    np_data = history.values
    return (np_data[:,5] - np_data[:,1]) / np_data[:,1]

def get_list_dates(symbol):
    filename = DATAROOT + symbol + ".csv"
    history = pd.read_csv(filename)
    np_data = history.values
    start = np_data[-1,0]
    end = np_data[0,0]
    return (start, end)

def main():
    symbols = os.listdir(DATAROOT)
    for symbol in symbols:
        print symbol
        print close_prices(symbol)

if __name__ == "__main__": main()
