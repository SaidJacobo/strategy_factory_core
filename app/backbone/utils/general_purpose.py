import logging
import cProfile
import pstats
import functools
import importlib.util
import sys
from functools import wraps
import contextvars


logger = logging.getLogger("general_purpose")

def load_function(dotpath: str):
    """Carga una función desde un módulo, recargando el módulo si ya ha sido importado."""
    module_, func = dotpath.rsplit(".", maxsplit=1)

    if module_ in sys.modules:
        # Si el módulo ya está en sys.modules, recargarlo
        spec = importlib.util.find_spec(module_)
        if spec is not None:
            m = importlib.util.module_from_spec(spec)
            sys.modules[module_] = m
            spec.loader.exec_module(m)
        else:
            raise ImportError(f"No se pudo encontrar el módulo: {module_}")
    else:
        # Importar normalmente si no ha sido cargado antes
        m = importlib.import_module(module_)

    return getattr(m, func)

screener_columns = [
    'industry',
    'sector',
    'trailingPE',
    'forwardPE',
    'pegRatio',
    'trailingPegRatio'
    'beta',
    'totalDebt',
    'quickRatio',
    'currentRatio',
    'totalRevenue',
    'debtToEquity',
    'revenuePerShare',
    'returnOnAssets',
    'returnOnEquity',
    'freeCashflow',
    'operatingCashflow',
    'earningsGrowth',
    'revenueGrowth',
    'bid',
    'ask',
    'marketCap',
    'twoHundredDayAverage',
    'recommendationKey',
    'numberOfAnalystOpinions',
    'symbol',
]


def transformar_a_uno(numero):
    # Inicializar contador de decimales
    decimales = 0
    while numero != int(numero):
        numero *= 10
        decimales += 1

    return 1 / (10 ** decimales)

def profile_function(output_file="code_performance_results.prof"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with cProfile.Profile() as profile:
                result = func(*args, **kwargs)
            stats = pstats.Stats(profile)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.dump_stats(output_file)
            return result
        return wrapper
    return decorator


# Contexto para saber si ya estamos dentro de un flujo
in_streaming_context = contextvars.ContextVar("in_streaming_context", default=False)

def streaming_endpoint(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        is_root = not in_streaming_context.get()
        if is_root:
            in_streaming_context.set(True)

        queue = kwargs.get("queue")

        try:
            return await func(*args, **kwargs)
        
        finally:
            if is_root and queue:
                await queue.put("DONE")
                in_streaming_context.set(False)  # restaurar estado
                
    return wrapper

def build_live_trading_config(used_backtests, risk_fn):
    config_file = {}

    for bt in used_backtests:
        strategy_name = bt.Bot.Strategy.Name
        ticker_name = bt.Bot.Ticker.Name

        if strategy_name not in config_file:
            config_file[strategy_name] = {
                'metatrader_name': bt.Bot.Strategy.MetaTraderName,
                'opt_params': None,
                'wfo_params': {
                    'use_wfo': False,
                    'look_back_bars': 200,
                    'warmup_bars': 200
                },
                'instruments_info': {}
            }

        if ticker_name not in config_file[strategy_name]['instruments_info']:
            config_file[strategy_name]['instruments_info'][ticker_name] = {}

        config_file[strategy_name]['instruments_info'][ticker_name]['risk'] = risk_fn(bt)
        config_file[strategy_name]['instruments_info'][ticker_name]['timeframe'] = bt.Bot.Timeframe.Name

    return config_file

def save_ticker_timeframes(config_file):
    ticker_timeframes = {}

    for strategy_data in config_file.values():
        instruments_info = strategy_data.get("instruments_info", {})
        for ticker, info in instruments_info.items():
            tf = info.get("timeframe")
            if ticker not in ticker_timeframes:
                ticker_timeframes[ticker] = set()
            ticker_timeframes[ticker].add(tf)

    # Convertimos los sets a listas antes de guardar
    ticker_timeframes_serializable = {
        ticker: sorted(list(timeframes))  # o list(timeframes) si no te importa el orden
        for ticker, timeframes in ticker_timeframes.items()
    }

    return ticker_timeframes_serializable