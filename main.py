import pandas as pd
import datetime
import DCC
import ARIMAObject
import stock_data_preprocessor as sdp

def main():
    # sdp.data_download()
    start = 1
    weekend_date = pd.date_range(start='2016-11-26',end=datetime.date.today())