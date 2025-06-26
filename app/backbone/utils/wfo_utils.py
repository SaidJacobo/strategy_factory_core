import os
from app.backbone.entities.ticker import Ticker
from app.backbone.entities.timeframe import Timeframe
from app.backbone.utils.general_purpose import transformar_a_uno
from unittest.mock import patch
from backtesting import Backtest
from backtesting.lib import FractionalBacktest
import pandas as pd
import plotly.express as px
from backtesting._stats import compute_stats
import numpy as np
from sklearn.linear_model import LinearRegression
import MetaTrader5 as mt5
from app.backbone.utils.get_data import get_data
import quantstats as qs
from scipy.stats import binom, jarque_bera, skew, kurtosis

np.seterr(divide="ignore")

def run_strategy(
    strategy,
    ticker: Ticker,
    timeframe: Timeframe,
    prices: pd.DataFrame,
    initial_cash: float,
    margin: float,
    risk_free_rate:float=0,
    risk=None,
    opt_params=None,
    metatrader_name=None,
    timezone=None,
):
    """
    Executes a backtest for a trading strategy with proper market metadata and commission handling.

    This function prepares the trading environment by:
    - Converting prices to the correct denomination
    - Loading symbol-specific trading constraints
    - Setting up commission structures
    - Selecting the appropriate backtest engine (fractional or standard)
    - Running the strategy with all configured parameters

    Steps performed:
    1. Converts prices using ticker-specific conversion rates
    2. Retrieves scaled symbol metadata (lot sizes, pip values, etc.)
    3. Configures commission calculation based on ticker category:
        - Absolute commission per contract for >=1
        - Percentage commission for <1
    4. Initializes either:
        - FractionalBacktest for fractional lot sizes
        - Standard Backtest for whole lot sizes
    5. Executes the strategy with all parameters and constraints
    6. Returns performance statistics and the backtest engine instance

    Parameters:
    - strategy: Trading strategy class/function to backtest
    - ticker (Ticker): Financial instrument being traded
    - timeframe (Timeframe): Time interval for the backtest
    - prices (pd.DataFrame): OHLC price data
    - initial_cash (float): Starting capital
    - margin (float): Margin requirement (1/leverage)
    - risk_free_rate (float, optional): Risk-free rate for Sharpe ratio. Defaults to 0.
    - risk (float, optional): Risk percentage per trade. Defaults to None.
    - opt_params (dict, optional): Optimization parameters. Defaults to None.
    - metatrader_name (str, optional): MT5 symbol name. Defaults to None.
    - timezone (str, optional): Timezone for trade timestamps. Defaults to None.

    Returns:
    - tuple: Contains:
        - stats: Backtest performance statistics (pd.Series/dict)
        - bt_train: The backtest engine instance (for further analysis)

    Side effects:
    - Makes MT5 API call to get symbol info (via mt5.symbol_info)
    - Modifies the input prices DataFrame with conversion rates
    - May log warnings about lot size rounding in backtest engine

    Notes:
    - Commission is applied both on entry and exit (hence /2 in calculation)
    - Fractional backtesting is automatically used when minimum_fraction < 1
    - All trading constraints (lot sizes, steps) come from broker metadata
    - The backtest engine handles spread as a fixed value from ticker.Spread
    - Returned stats typically include Sharpe ratio, drawdown, trade counts etc.
    - The bt_train object can be used to access trade-by-trade details
    """

    prices = get_conversion_rate(prices, ticker, timeframe)
    
    (
        scaled_pip_value,
        scaled_minimum_lot,
        scaled_maximum_lot,
        scaled_contract_volume,
        minimum_fraction,
        volume_step,
    ) = get_scaled_symbol_metadata(ticker.Name)

    bt_train = None
    info = mt5.symbol_info(ticker.Name)

    if ticker.Category.Commission >= 1:
        commission = lambda size, price: abs(size) * (ticker.Category.Commission / 2) / info.trade_contract_size
    else:
        commission = lambda size, price: abs(size) * price * ((ticker.Category.Commission / 2) / 100)

    if minimum_fraction < 1:
        bt_train = FractionalBacktest(
            prices, 
            strategy,
            commission=commission, 
            cash=initial_cash, 
            margin=margin,
            fractional_unit=minimum_fraction,
            spread=ticker.Spread
        )

    else:
        bt_train = Backtest(
            prices, 
            strategy,
            commission=commission, 
            cash=initial_cash, 
            margin=margin, 
            spread=ticker.Spread
        )

    stats = bt_train.run(
        risk_free_rate=risk_free_rate,
        pip_value=scaled_pip_value,
        minimum_lot=scaled_minimum_lot,
        maximum_lot=scaled_maximum_lot,
        contract_volume=scaled_contract_volume,
        volume_step=volume_step,
        risk=risk,
        ticker=ticker,
        opt_params=opt_params,
        metatrader_name=metatrader_name,
        timezone=timezone,
        minimum_fraction=minimum_fraction
    )

    return stats, bt_train

def run_strategy_and_get_performances(
    strategy,
    ticker: Ticker,
    timeframe: Timeframe,
    prices: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float,
    margin: float,
    risk=None,
    plot_path=None,
    file_name=None,
    opt_params=None,
    save_report=False
):
    
    """
    Executes a trading strategy backtest and computes comprehensive performance metrics,
    with optional visualization and reporting capabilities.

    This function extends the basic backtest by:
    - Generating detailed performance statistics
    - Calculating advanced metrics (stability ratio, Jarque-Bera, etc.)
    - Producing visualizations and HTML reports
    - Segmenting trade analytics (long/short, winning/losing trades)
    - Computing risk-adjusted return metrics

    Steps performed:
    1. Runs the core strategy backtest using run_strategy()
    2. Generates equity curve plots if plot_path is specified
    3. Creates QuantStats performance reports if save_report=True
    4. Computes trade-level metrics:
        - Percentage returns relative to account equity
        - Trade durations in days
        - Win/loss segmentation
    5. Calculates advanced statistics:
        - Equity curve stability ratio (linear regression R²)
        - Winrate binomial p-value
        - Return distribution metrics (skew, kurtosis)
    6. Compiles results into three structured DataFrames:
        - Strategy-level performance metrics
        - Detailed trade performance analytics
        - Raw backtest statistics

    Parameters:
    - strategy: Trading strategy implementation
    - ticker (Ticker): Financial instrument configuration
    - timeframe (Timeframe): Backtesting time interval
    - prices (pd.DataFrame): OHLC price data
    - initial_cash (float): Starting capital
    - risk_free_rate (float): Risk-free rate for Sharpe ratio
    - margin (float): Margin requirement (1/leverage)
    - risk (float, optional): Risk percentage per trade. Default=None.
    - plot_path (str, optional): Directory to save plots. Default=None.
    - file_name (str, optional): Base name for output files. Default=None.
    - opt_params (dict, optional): Optimization parameters. Default=None.
    - save_report (bool, optional): Whether to save QuantStats report. Default=False.

    Returns:
    - tuple: Three DataFrames containing:
        - df_stats (pd.DataFrame): Strategy performance metrics (1 row)
        - trade_performance (pd.DataFrame): Aggregated trade analytics (1 row)
        - stats: Raw backtest statistics object

    Side effects:
    - Creates plot files in plot_path if specified:
        - Interactive equity curve plot (.html)
        - QuantStats performance report (if save_report=True)
    - May create directories if they don't exist

    Notes:
    - Trade returns are calculated as percentage of equity at entry
    - Duration is converted to whole days for consistency
    - Stability ratio measures equity curve linearity (higher = smoother)
    - Winrate p-value tests if winrate could occur by chance
    - Jarque-Bera tests return distribution normality
    - All metrics are rounded to 3 decimal places
    - Missing values are filled with 0 for robustness
    - Separate metrics are provided for long/short positions
    """
    
    stats, bt_train = run_strategy(
        strategy=strategy,
        ticker=ticker,
        timeframe=timeframe,
        prices=prices,
        initial_cash=initial_cash,
        risk_free_rate=risk_free_rate,
        margin=margin,
        risk=risk,
        opt_params=opt_params,
    )

    if plot_path:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
            
        bt_train.plot(
            filename=os.path.join(plot_path, file_name + '.html'), 
            resample=False, 
            open_browser=False
        )

        if save_report:
            returns = stats._equity_curve['Equity'].pct_change().dropna()
            qs.reports.html(
                returns, 
                output=os.path.join(plot_path, 'reports', file_name + '.html'), 
                title=file_name, 
                rf=risk_free_rate
            )

    equity_curve = stats._equity_curve
    trades = stats._trades
    
    trades = pd.merge(
        trades,
        equity_curve['Equity'],
        left_on='ExitTime',
        right_index=True,
        how='inner'
    )
    
    trades['ReturnPct'] = (trades['NetPnL'] / trades['Equity'].shift(1)) * 100
    if len(trades) > 0:
        trades.loc[0, 'ReturnPct'] = (trades.loc[0, 'NetPnL'] / initial_cash) * 100

    trades['Duration'] = pd.to_timedelta(trades['Duration'])
    trades['Duration'] = (trades['Duration'].dt.total_seconds() // 3600 // 24).astype(int)
      
    stats._trades = trades.round(5)
    
    winning_trades = trades[trades["NetPnL"]>=0]
    losing_trades = trades[trades["NetPnL"]<0]

    long_trades = trades[trades["Size"] >= 0]
    short_trades = trades[trades["Size"] < 0]
    
    long_winning_trades = long_trades[long_trades["NetPnL"] >= 0]
    long_losing_trades = long_trades[long_trades["NetPnL"] < 0]
    
    short_winning_trades = short_trades[short_trades["NetPnL"] >= 0]
    short_losing_trades = short_trades[short_trades["NetPnL"] < 0]
    
    equity_curve = equity_curve["Equity"].values
    
    x = np.arange(len(equity_curve)).reshape(-1, 1)
    reg = LinearRegression().fit(x, equity_curve)
    stability_ratio = reg.score(x, equity_curve)

    stats["Duration"] = pd.to_timedelta(stats["Duration"])

    winrate_p_value = calculate_binomial_p_value(
        n=trades.shape[0], 
        k=winning_trades.shape[0]
    )

    returns = trades['Equity'].pct_change().dropna()  # Elimina NaN del primer valor

    jb_stat, jb_p_value = jarque_bera(returns)
    skew_value = skew(returns)
    kurtosis_value = kurtosis(returns, fisher=True)  # True para exceso sobre normal

    df_stats = pd.DataFrame(
        {
            "StabilityRatio": [stability_ratio],
            "Trades": [stats["# Trades"]],
            "Return": [stats["Return [%]"]],
            "Drawdown": [np.abs(stats["Max. Drawdown [%]"])],
            "RreturnDd": [stats["Return [%]"] / np.abs(stats["Max. Drawdown [%]"])],
            "WinRate": [stats["Win Rate [%]"]],
            "Duration": [stats["Duration"].days],

            "ExposureTime": [stats["Exposure Time [%]"]],
            "KellyCriterion": [stats["Kelly Criterion"]],
            "WinratePValue": [winrate_p_value],
            "SharpeRatio": [stats["Sharpe Ratio"]],

            "JarqueBeraStat": [jb_stat],
            "JarqueBeraPValue": [jb_p_value],
            "Skew": [skew_value],
            "Kurtosis": [kurtosis_value],

        }
    )
    
    df_stats["StabilityWeightedRar"] = (df_stats["Return"] / (1 + df_stats["Drawdown"])) * np.log(1 + df_stats["Trades"]) * stability_ratio
    df_stats = df_stats.fillna(0).round(3)

    consecutive_wins, consecutive_losses = max_consecutive_wins_and_losses(trades)
    
    trade_performance = pd.DataFrame(
        {
            # General
            "MeanReturnPct":[trades.ReturnPct.mean()],
            "StdReturnPct":[trades.ReturnPct.std()],
            "MeanTradeDuration":[trades['Duration'].mean()],
            "StdTradeDuration":[trades['Duration'].std()],

            "MeanWinningReturnPct":[winning_trades.ReturnPct.mean()],
            "StdWinningReturnPct":[winning_trades.ReturnPct.std()],

            "MeanLosingReturnPct":[losing_trades.ReturnPct.mean()],
            "StdLosingReturnPct":[losing_trades.ReturnPct.std()],

            # Longs
            "LongWinrate": [(long_winning_trades.size / long_trades.size) * 100 if long_trades.size > 0 else 0],
            "LongMeanReturnPct": [long_trades.ReturnPct.mean()],
            "LongStdReturnPct": [long_trades.ReturnPct.std()],
            
            "WinLongMeanReturnPct": [long_winning_trades.ReturnPct.mean()],
            "WinLongStdReturnPct": [long_winning_trades.ReturnPct.std()],
            "LoseLongMeanReturnPct": [long_losing_trades.ReturnPct.mean()],
            "LoseLongStdReturnPct": [long_losing_trades.ReturnPct.std()],
            
            # Shorts
            "ShortWinrate": [(short_winning_trades.size / short_trades.size) * 100 if short_trades.size > 0 else 0],
            "ShortMeanReturnPct": [short_trades.ReturnPct.mean()],
            "ShortStdReturnPct": [short_trades.ReturnPct.std()],
            
            "WinShortMeanReturnPct": [short_winning_trades.ReturnPct.mean()],
            "WinShortStdReturnPct": [short_winning_trades.ReturnPct.std()],
            "LoseShortMeanReturnPct": [short_losing_trades.ReturnPct.mean()],
            "LoseShortStdReturnPct": [short_losing_trades.ReturnPct.std()],

            # Otras metricas
            "ProfitFactor": [stats["Profit Factor"]],
            "WinRate": [stats["Win Rate [%]"]],
            "ConsecutiveWins": [consecutive_wins],
            "ConsecutiveLosses": [consecutive_losses],
            "LongCount": [long_trades.shape[0]],
            "ShortCount": [short_trades.shape[0]],
        }
    ).fillna(0)

    return df_stats, trade_performance.round(3), stats

def get_conversion_rate(prices: pd.DataFrame, ticker: Ticker, timeframe: Timeframe):
    """
    Calculates and applies currency conversion rates to price data for non-USD denominated instruments.

    This function handles currency conversion for Forex, Metals, Crypto, and Exotics by:
    - Identifying if the instrument needs conversion (non-USD or inverse USD pairs)
    - Finding the appropriate USD-based counterpart pair
    - Applying direct or inverse rates as needed
    - Merging conversion rates with the original price data

    Steps performed:
    1. Checks if the ticker category requires conversion (Forex, Metals, Crypto, Exotics)
    2. For non-USD pairs (e.g., EURGBP):
       - Attempts to find the USD-quoted version (GBPUSD)
       - Falls back to inverse pair (USDGBP) if direct not available
       - Calculates inverse rates when needed
    3. For USD-prefixed pairs (e.g., USDJPY):
       - Applies direct inverse (1/USDJPY)
    4. For non-convertible categories:
       - Sets conversion rate to 1 (no conversion)
    5. Merges conversion rates with original prices via forward-fill

    Parameters:
    - prices (pd.DataFrame): OHLC price data with DateTime index
    - ticker (Ticker): Instrument information including:
        - Name (e.g., 'EURGBP', 'USDJPY')
        - Category (determines if conversion needed)
    - timeframe (Timeframe): Used to fetch conversion rates at matching intervals

    Returns:
    - pd.DataFrame: Original prices with added 'ConversionRate' column:
        - 1.0 for non-convertible instruments
        - Direct rate for USD-prefixed pairs
        - Cross-calculated rate for other Forex pairs

    Raises:
    - Exception: When no valid conversion pair can be found for a non-USD instrument

    Notes:
    - Conversion rates are forward-filled to handle mismatched timestamps
    - Always targets USD conversion (assumes USD is account currency)
    - For pairs like EURGBP, conversion goes through GBPUSD first
    - Metals (XAUUSD) and Crypto (BTCUSD) follow same logic as Forex
    - The function preserves all original price columns
    """
    categories = ['Forex', 'Metals', 'Crypto', 'Exotics'] # <-- Ajustar
    
    if ticker.Category.Name in categories:
        date_from = prices.index[0]
        date_to = prices.index[-1]
        
        if 'USD' not in ticker.Name: # Por ejemplo CHFNZD
            # En este caso tengo que buscar idealmente NZDUSD 
            quoted_currency = ticker.Name[3:]

            # Caso ideal
            asosiated_ticker = quoted_currency + 'USD' # <-- aca deberia ir la divisa de la cuenta
            usd_prices = get_data(asosiated_ticker, timeframe.MetaTraderNumber, date_from, date_to)

            if usd_prices.empty:
                asosiated_ticker = 'USD' + quoted_currency
                usd_prices = get_data(asosiated_ticker, timeframe.MetaTraderNumber, date_from, date_to)

                if usd_prices.empty:
                    raise Exception("Can't calculate Conversion rate")

                usd_prices['Open'] = 1 / usd_prices['Open']

            usd_prices = usd_prices[['Open']]
            usd_prices = usd_prices.rename(columns={'Open':'ConversionRate'})
            usd_prices.index = pd.to_datetime(usd_prices.index)

            prices = pd.merge(
                left=prices,
                right=usd_prices,
                how='left',
                right_index=True,
                left_index=True
            ).ffill()
            
        elif ticker.Name.startswith('USD'):
            prices['ConversionRate'] = 1 / prices['Open']
    
    else:
        prices['ConversionRate'] = 1
        
    return prices

def optimization_function(stats):
    return (
        (stats["Return [%]"] / (1 + (-1 * stats["Max. Drawdown [%]"])))
    )

def plot_full_equity_curve(df_equity, title):

    fig = px.line(x=df_equity.index, y=df_equity.Equity)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Equity")
    fig.update_traces(textposition="bottom right")
    fig.show()

def get_scaled_symbol_metadata(ticker: str, metatrader=None):

    if metatrader:
        info = metatrader.symbol_info(ticker)
    else:
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()
        info = mt5.symbol_info(ticker)
    contract_volume = info.trade_contract_size
    minimum_lot = info.volume_min
    maximum_lot = info.volume_max
    pip_value = info.trade_tick_size
    minimum_units = contract_volume * minimum_lot
    volume_step = info.volume_step

    minimum_fraction = transformar_a_uno(minimum_units)

    scaled_contract_volume = contract_volume / minimum_fraction

    scaled_pip_value = pip_value * minimum_fraction
    scaled_minimum_lot = minimum_lot / minimum_fraction
    scaled_maximum_lot = maximum_lot / minimum_fraction

    return (
        scaled_pip_value,
        scaled_minimum_lot,
        scaled_maximum_lot,
        scaled_contract_volume,
        minimum_fraction,
        volume_step
    )

def calculate_binomial_p_value(n, k, p=0.5):
    """
    Calcula el p-valor para la hipótesis nula de que la probabilidad de ganar es p (por defecto 0.5),
    dado que se observaron k trades ganadores en n trades.

    Retorna la probabilidad de obtener al menos k éxitos por azar (cola superior).
    """
    if k > n:
        raise ValueError("k no puede ser mayor que n")
    
    p_valor = 1 - binom.cdf(k - 1, n, p)
    return round(p_valor, 3)


def max_consecutive_wins_and_losses(df_trades: pd.DataFrame):
    trades_copy = df_trades.copy()

    # Create a boolean column to identify if it was a winning trade
    trades_copy['win'] = trades_copy['PnL'] > 0

    # Detect changes in streaks (win -> loss or loss -> win)
    trades_copy['streak_id'] = (trades_copy['win'] != trades_copy['win'].shift()).cumsum()

    # Group by each streak and count its length
    streaks = trades_copy.groupby(['streak_id', 'win']).size().reset_index(name='duration')

    # Filter and count how many streaks of each type exist
    winning_streaks = streaks[streaks['win'] == True]
    losing_streaks = streaks[streaks['win'] == False]

    max_consecutive_wins = winning_streaks['duration'].max() if not winning_streaks.empty else 0
    max_consecutive_losses = losing_streaks['duration'].max() if not losing_streaks.empty else 0

    return max_consecutive_wins, max_consecutive_losses

def walk_forward(
    strategy,
    data_full,
    warmup_bars,
    lookback_bars=28 * 1440,
    validation_bars=7 * 1440,
    params=None,
    cash=15_000,
    commission=0.0002,
    margin=1 / 30,
    verbose=False,
):

    optimized_params_history = {}
    stats_master = []
    equity_final = None

    # Iniciar el índice en el final del primer lookback

    i = lookback_bars + warmup_bars

    while i < len(data_full):

        train_data = data_full.iloc[i - lookback_bars - warmup_bars : i]

        if verbose:
            print(f"train from {train_data.index[0]} to {train_data.index[-1]}")
        bt_training = Backtest(
            train_data, strategy, cash=cash, commission=commission, margin=margin
        )

        with patch("backtesting.backtesting._tqdm", lambda *args, **kwargs: args[0]):
            stats_training = bt_training.optimize(**params)
        remaining_bars = len(data_full) - i
        current_validation_bars = min(validation_bars, remaining_bars)

        validation_data = data_full.iloc[i - warmup_bars : i + current_validation_bars]

        validation_date = validation_data.index[warmup_bars]

        if verbose:
            print(f"validate from {validation_date} to {validation_data.index[-1]}")
        bt_validation = Backtest(
            validation_data,
            strategy,
            cash=cash if equity_final is None else equity_final,
            commission=commission,
            margin=margin,
        )

        validation_params = {
            param: getattr(stats_training._strategy, param)
            for param in params.keys()
            if param != "maximize"
        }

        optimized_params_history[validation_date] = validation_params

        if verbose:
            print(validation_params)
        stats_validation = bt_validation.run(**validation_params)

        equity_final = stats_validation["Equity Final [$]"]

        if verbose:
            print(f"equity final: {equity_final}")
            print("=" * 32)
        stats_master.append(stats_validation)

        # Mover el índice `i` al final del período de validación actual

        i += current_validation_bars
    wfo_stats = get_wfo_stats(stats_master, warmup_bars, data_full)

    return wfo_stats, optimized_params_history

def get_wfo_stats(stats, warmup_bars, ohcl_data):
    trades = pd.DataFrame(
        columns=[
            "Size",
            "EntryBar",
            "ExitBar",
            "EntryPrice",
            "ExitPrice",
            "PnL",
            "ReturnPct",
            "EntryTime",
            "ExitTime",
            "Duration",
        ]
    )
    for stat in stats:
        trades = pd.concat([trades, stat._trades])
    trades.EntryBar = trades.EntryBar.astype(int)
    trades.ExitBar = trades.ExitBar.astype(int)

    equity_curves = pd.DataFrame(columns=["Equity", "DrawdownPct", "DrawdownDuration"])
    for stat in stats:
        equity_curves = pd.concat(
            [equity_curves, stat["_equity_curve"].iloc[warmup_bars:]]
        )
    wfo_stats = compute_stats(
        trades=trades,  # broker.closed_trades,
        equity=equity_curves.Equity,
        ohlc_data=ohcl_data,
        risk_free_rate=0.0,
        strategy_instance=None,  # strategy,
    )

    wfo_stats["_equity"] = equity_curves
    wfo_stats["_trades"] = trades

    return wfo_stats

def run_wfo(
    strategy,
    ticker,
    interval,
    prices: pd.DataFrame,
    initial_cash: float,
    commission: float,
    margin: float,
    optim_func,
    params: dict,
    lookback_bars: int,
    warmup_bars: int,
    validation_bars: int,
    plot=True,
    risk:None=float,
):

    (
        scaled_pip_value,
        scaled_minimum_lot,
        scaled_maximum_lot,
        scaled_contract_volume,
        minimum_fraction,
        trade_tick_value_loss,
        volume_step
    ) = get_scaled_symbol_metadata(ticker)

    scaled_prices = prices.copy()
    scaled_prices.loc[:, ["Open", "High", "Low", "Close"]] = (
        scaled_prices.loc[:, ["Open", "High", "Low", "Close"]].copy() * minimum_fraction
    )

    params["minimum_lot"] = [scaled_minimum_lot]
    params["maximum_lot"] = [scaled_maximum_lot]
    params["contract_volume"] = [scaled_contract_volume]
    params["pip_value"] = [scaled_pip_value]
    params["trade_tick_value_loss"] = [trade_tick_value_loss]
    params["volume_step"] = [volume_step]
    params["risk"] = [risk]

    params["maximize"] = optim_func

    wfo_stats, optimized_params_history = walk_forward(
        strategy,
        scaled_prices,
        lookback_bars=lookback_bars,
        validation_bars=validation_bars,
        warmup_bars=warmup_bars,
        params=params,
        commission=commission,
        margin=margin,
        cash=initial_cash,
        verbose=False,
    )

    df_equity = wfo_stats["_equity"]
    df_trades = wfo_stats["_trades"]

    if plot:
        plot_full_equity_curve(df_equity, title=f"{ticker}, {interval}")
    # Calculo el stability ratio

    x = np.arange(df_equity.shape[0]).reshape(-1, 1)
    reg = LinearRegression().fit(x, df_equity.Equity)
    stability_ratio = reg.score(x, df_equity.Equity)

    # Extraigo metricas

    df_stats = pd.DataFrame(
        {
            "strategy": [strategy.__name__],
            "ticker": [ticker],
            "interval": [interval],
            "stability_ratio": [stability_ratio],
            "return": [wfo_stats["Return [%]"]],
            "final_eq": [wfo_stats["Equity Final [$]"]],
            "drawdown": [wfo_stats["Max. Drawdown [%]"]],
            "drawdown_duration": [wfo_stats["Max. Drawdown Duration"]],
            "win_rate": [wfo_stats["Win Rate [%]"]],
            "sharpe_ratio": [wfo_stats["Sharpe Ratio"]],
            "trades": [df_trades.shape[0]],
            "avg_trade_percent": [wfo_stats["Avg. Trade [%]"]],
            "exposure": [wfo_stats["Exposure Time [%]"]],
            "final_equity": [wfo_stats["Equity Final [$]"]],
            "Duration": [wfo_stats["Duration"]],
        }
    )

    return wfo_stats, df_stats, optimized_params_history
