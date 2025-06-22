import pandas as pd
import pandas_ta as pandas_ta
import MetaTrader5 as mt5
import MetaTrader5 as mt5
import pandas as pd

import random
random.seed(42)

def get_data(
        ticker, 
        timeframe, 
        date_from, 
        date_to, 
        save_in=None
    ):

    print("MetaTrader5 package author: ", mt5.__author__)
    print("MetaTrader5 package version: ", mt5.__version__)

    # Establecer conexión con el terminal de MetaTrader 5
    if not mt5.initialize():
        raise Exception("initialize() failed, error code =", mt5.last_error())

    rates = mt5.copy_rates_range(ticker, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    prices_df = pd.DataFrame(rates)

    # Convertir el tiempo de segundos a formato datetime
    prices_df['time'] = pd.to_datetime(prices_df['time'], unit='s')

    # Renombrar columnas para el ticker principal
    prices_df = prices_df.rename(columns={
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume'
    }).set_index('Date')

    prices_df.index = prices_df.index.tz_localize('UTC').tz_convert('UTC')

    return prices_df