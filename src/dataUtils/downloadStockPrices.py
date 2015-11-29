import wget
import sys

start = ("11", "12", "1980")
end = ("8", "22", "2015")

def getStockCsv(symbol):
    queryStr = "http://real-chart.finance.yahoo.com/table.csv?"
    queryStr += "s="+symbol
    queryStr += "&a={}&b={}&c={}&d={}&e={}&f={}&g=d&ignore=.csv".format(start[0], start[1], start[2], end[0], end[1], end[2])
    filename = wget.download(queryStr, symbol + ".csv")


def main(argFile):
    with open(argFile, "r") as inFile:
        for line in inFile:
            symbol = line.split('|')[0]
            print 'getting stock data for: ', symbol
            getStockCsv(symbol)

if __name__ == "__main__":
       main(sys.argv[1])
