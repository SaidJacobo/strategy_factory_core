from collections import namedtuple
from typing import List
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
import yaml
from app.backbone.entities.bot_performance import BotPerformance
from app.backbone.entities.trade import Trade
import pandas as pd
import uuid

# Los ejemplos para estos test se pueden encontrar aqui:
# https://docs.google.com/spreadsheets/d/10OSCpBoGuY5uuCzZcLtCrbHPkMUrr1uSpLN8SDbRbH8/edit?gid=2082342944#gid=2082342944


def _performance_from_df_to_obj(
    df_performance: DataFrame, 
    date_from, 
    date_to, 
    risk, 
    method, 
    bot, 
    initial_cash, 
    ):
    performance_for_db = [BotPerformance(**row) for _, row in df_performance.iterrows()].pop()
    performance_for_db.DateFrom = date_from
    performance_for_db.DateTo = date_to
    performance_for_db.Risk = risk
    performance_for_db.Method = method
    performance_for_db.Bot = bot
    performance_for_db.InitialCash = initial_cash
    
    return performance_for_db

def get_trade_df_from_db(trades: List[Trade], performance_id=None):
    # Obtener nombres de columnas directamente desde el modelo
    columns = [col.name for col in Trade.__table__.columns if col.name != 'BotPerformance']

    if not trades:
        return pd.DataFrame(columns=columns)

    # Construir los dicts con getattr dinámicamente
    data = [
        {col: getattr(trade, col) for col in columns}
        for trade in trades
    ]

    df = pd.DataFrame(data)
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    df = df.sort_values(by='ExitTime')
    df['Date'] = df['ExitTime']
    df.set_index('Date', inplace=True)

    return df

def trades_from_df_to_obj(trades: pd.DataFrame) -> List[Trade]:
    trade_columns = [column for column in trades.columns if hasattr(Trade, column)]
    trades_db = [Trade(**row[trade_columns]) for _, row in trades.iterrows()]

    return trades_db

def get_date_range(equity_curves: pd.DataFrame):
    min_date = None
    max_date = None

    for name, curve in equity_curves.items():

        if curve.empty:
            continue
        # Convertir las fechas a UTC si son tz-naive
        actual_date = curve.index[0].tz_localize('UTC') if curve.index[0].tz is None else curve.index[0].tz_convert('UTC')
        
        # Si min_date es None, inicializar con la primera fecha
        if min_date is None:
            min_date = actual_date
        # Comparar si la fecha actual es menor que min_date
        elif actual_date < min_date:
            min_date = actual_date

        # Si max_date es None, inicializar con la última fecha
        curve_last_date = curve.index[-1].tz_localize('UTC') if curve.index[-1].tz is None else curve.index[-1].tz_convert('UTC')
        
        if max_date is None:
            max_date = curve_last_date
        # Comparar si la fecha actual es mayor que max_date
        elif curve_last_date > max_date:
            max_date = curve_last_date

    # Calcular min_date y max_date
    min_date = min_date.date()
    max_date = max_date.date()

    date_range = pd.to_datetime(pd.date_range(start=min_date, end=max_date, freq='D'))
    return date_range

def calculate_stability_ratio(equity_curve: pd.Series):
    x = np.arange(len(equity_curve)).reshape(-1, 1)
    reg = LinearRegression().fit(x, equity_curve)
    stability_ratio = reg.score(x, equity_curve)
    
    return stability_ratio

def max_drawdown(equity_curve, verbose=True):
    # Calcular el running max de la equity curve
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calcular el drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Encontrar el valor máximo de drawdown y la fecha correspondiente
    max_drawdown_value = np.min(drawdown) * 100  # Convertir el drawdown a porcentaje
    max_drawdown_date = equity_curve.index[np.argmin(drawdown)]
    
    if verbose:
        print(f"Máximo drawdown: {max_drawdown_value:.2f}%")
        print(f"Fecha del máximo drawdown: {max_drawdown_date}")

    return max_drawdown_value

def calculate_sharpe_ratio(returns, risk_free_rate=None, trading_periods=252):
    excess_returns = returns - (risk_free_rate / trading_periods)
    sharpe = excess_returns.mean() / returns.std()
    return sharpe * np.sqrt(trading_periods)