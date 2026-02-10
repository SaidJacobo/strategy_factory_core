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

def get_portfolio_equity_curve(all_bot_trades: dict[str, pd.DataFrame], initial_equity: float) -> pd.Series:
    
    date_range = get_date_range(equity_curves=all_bot_trades)
    trades = pd.DataFrame()

    for bot_name, bot_trades in all_bot_trades.items():
        if not bot_trades.empty:
            trades = pd.concat([trades, bot_trades])

    trades['EntryTime'] = pd.to_datetime(trades['EntryTime'])
    trades['ExitTime'] = pd.to_datetime(trades['ExitTime'])

    # Generate unique IDs for each trade
    trades['UniqueId'] = [str(uuid.uuid4()) for _ in range(len(trades))]
    
    opens = trades[['UniqueId', 'EntryTime']].rename(columns={'EntryTime':'EventDate'})

    closes = trades[['UniqueId', 'ExitTime', 'ReturnPct']].rename(columns={'ExitTime':'EventDate'})

    events = pd.concat([opens, closes])

    events['ReturnOrder'] = (events['ReturnPct'] != 0).astype(int)  # 0 si ReturnPct es 0, 1 si es distinto de 0
    events = events.sort_values(by=['EventDate', 'ReturnOrder'], ascending=[True, True]).fillna(0).reset_index().drop(columns=['Date'])
    events = events.drop(columns=['ReturnOrder'])  # Eliminar la columna auxiliar

    events['Equity'] = np.nan

    events.iloc[0, events.columns.get_loc('Equity')] = initial_equity
    
    for index, row in events.iterrows():

        if row.ReturnPct != 0:

            open_trade = events[(events.UniqueId == row.UniqueId) & (events.ReturnPct == 0)].iloc[0]

            last_equity = events.iloc[index - 1, events.columns.get_loc('Equity')]

            events.iloc[index, events.columns.get_loc('Equity')] = last_equity + (open_trade.Equity * (row.ReturnPct / 100))

        else:
            last_equity = events.iloc[index - 1, events.columns.get_loc('Equity')] if index > 0 else initial_equity
            
            events.iloc[index, events.columns.get_loc('Equity')] = last_equity
    
    events['EventDate'] = events['EventDate'].dt.floor('D').dt.date
    events = events.groupby('EventDate').agg({'Equity':'last'})
    events = events.reindex(date_range)
    events.Equity = events.Equity.ffill()

    return events[['Equity']]

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

FtmoChallengeMetrics = namedtuple('FtmoChallengeMetrics',
    [
        'positive_hits',
        'negative_hits',
        'success_ratio',
        'mean_time_to_positive', 
        'std_time_to_positive',
        'mean_time_to_negative',
        'std_time_to_negative',
    ]
)

def ftmo_simulator(equity_curve: np.array, initial_cash, positive_hit_threshold, negative_hit_threshold):
    def safe_mean(arr):
        return np.mean(arr) if len(arr) > 0 else 0

    def safe_std(arr):
        return np.std(arr) if len(arr) > 0 else 0

    total_positive_hits = 0
    total_negative_hits = 0
    all_time_to_positive = []
    all_time_to_negative = []

    for i in range(0, len(equity_curve)):
        perc_change = 0
        time_to_positive = []
        time_to_negative = []

        actual_equity = equity_curve[i]

        days_elapsed = 0

        for j in range(i, len(equity_curve)):
            future_equity = equity_curve[j]  # Acceso directo en numpy es más rápido

            if i == 0 and j == 0:
                perc_change = ((future_equity - initial_cash) / initial_cash) * 100
            else:
                perc_change = ((future_equity - actual_equity) / actual_equity) * 100

            days_elapsed += 1

            if perc_change >= positive_hit_threshold:
                total_positive_hits += 1
                time_to_positive.append(days_elapsed)
                days_elapsed = 0
                break

            elif perc_change <= -negative_hit_threshold:
                total_negative_hits += 1
                time_to_negative.append(days_elapsed)
                days_elapsed = 0
                break

        all_time_to_positive.extend(time_to_positive)
        all_time_to_negative.extend(time_to_negative)

    total_hits = total_positive_hits + total_negative_hits
    ftmo_challenge_metrics = FtmoChallengeMetrics(
    negative_hits=total_negative_hits,
    positive_hits=total_positive_hits,
    success_ratio=round(total_positive_hits / (total_hits), 3) if total_hits > 0 else 0,
    mean_time_to_positive=round(safe_mean(all_time_to_positive), 3),
    mean_time_to_negative=round(safe_mean(all_time_to_negative), 3),
    std_time_to_positive=round(safe_std(all_time_to_positive), 3),
    std_time_to_negative=round(safe_std(all_time_to_negative), 3),
    )

    return ftmo_challenge_metrics

MarginMetrics = namedtuple('MarginMetrics',
    [
        'margin_calls',
        'stop_outs',
    ]
)
def calculate_margin_metrics(all_trades, portfolio_equity_curve):

    # Convertir columnas a datetime
    with open("./app/configs/leverages.yml", "r") as file_name:
        leverages = yaml.safe_load(file_name)
    
    for bot_name, trades in all_trades.items():

        ticker = bot_name.split('_')[1]
        leverage = leverages[ticker]
        
        trades["EntryTime"] = pd.to_datetime(trades["EntryTime"])
        trades["ExitTime"] = pd.to_datetime(trades["ExitTime"])
        
        trades['margin'] = (np.abs(trades['Size']) * trades['EntryPrice'] * trades['EntryConversionRate']) / leverage
        

    # Concatenar y calcular los eventos
    all_events = pd.concat([
        pd.concat([
            df[["EntryTime", "margin"]].rename(columns={"EntryTime": "time", "margin": "change"}).round(3),
            df[["ExitTime", "margin"]].rename(columns={"ExitTime": "time", "margin": "change"}).assign(change=lambda x: -x["change"]).round(3)
        ]) for df in all_trades.values()
    ])

    # Ordenar por tiempo
    all_events = all_events.sort_values(['time', 'change'], ascending=[True, False]).reset_index(drop=True)

    # Calcular el margen acumulado
    all_events["margin"] = all_events["change"].cumsum()

    all_events['time'] = pd.to_datetime(all_events['time']).dt.date
    all_events.set_index('time', inplace=True)

    all_events = pd.merge(
        all_events,
        portfolio_equity_curve,
        left_index=True,
        right_index=True,
        how='left'
    )

    all_events = all_events.round(2)
    
    all_events['margin_level'] = ((all_events['Equity'] / all_events['margin']) * 100)

    all_events['free_margin'] = (all_events['Equity'] - all_events['margin'])

    stop_outs = all_events[(all_events['margin_level'] < 50) & (all_events['margin_level'] != -1*np.inf)]
    margin_calls = all_events[(all_events['margin_level'] < 100) & (all_events['margin_level'] != -1*np.inf)]
    
    margin_metrics = MarginMetrics(
        margin_calls=margin_calls.shape[0], 
        stop_outs=stop_outs.shape[0]
    )
    
    return margin_metrics

def get_portfolio_equity_differences(equity_curves, initial_cash) -> dict:
    differences = {}
    for name, curve in equity_curves.items():
        equity_df = curve.copy()
        
        equity_df = equity_df.reset_index().rename(columns={'index':'Date'})
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        equity_df['Date'] = equity_df['Date'].dt.floor('D').dt.date
        
        date_range = get_date_range(equity_curves=equity_curves)

        equity_df = pd.DataFrame(equity_df.groupby('Date')['Equity'].last()).reindex(date_range)
        equity_df.loc[:, :] = equity_df.ffill()
        equity_df.loc[:, :] = equity_df.fillna(initial_cash)

        equity_df['diff'] = equity_df['Equity'] - equity_df['Equity'].shift(1)
        
        differences[name] = equity_df.fillna(0)
    
    return differences

def get_portfolio_correlation_matrix(differences: dict) -> pd.DataFrame:
    all_equity_df = pd.DataFrame()

    for name, df in differences.items():
        all_equity_df[name] = df.resample('ME').agg({'Equity':'last','diff':'sum',})['diff']

    correlation_matrix = all_equity_df.corr(method='pearson')

    return correlation_matrix

def calculate_sharpe_ratio(returns, risk_free_rate=None, trading_periods=252):
    excess_returns = returns - (risk_free_rate / trading_periods)
    sharpe = excess_returns.mean() / returns.std()
    return sharpe * np.sqrt(trading_periods)