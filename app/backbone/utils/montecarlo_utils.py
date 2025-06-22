import numpy as np
import pandas as pd

def max_drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return np.min(drawdown) * 100  # Convertir el drawdown a porcentaje


def montecarlo_statistics_simulation(
    trade_history,
    equity_curve,
    n_simulations,
    initial_equity,
    threshold_ruin=0.85,
    return_raw_curves=False,
    percentiles=None,
):

    # Renombro las columnas

    trade_history = trade_history.rename(columns={"ExitTime": "Date"})
    trade_history = trade_history[["Date", "NetPnL"]]

    equity_curve = (
        equity_curve.reset_index()
        .rename(columns={"index": "Date"})[["Date", "Equity"]]
        .sort_values(by="Date")
    )

    trade_history["Date"] = pd.to_datetime(trade_history["Date"])
    equity_curve["Date"] = pd.to_datetime(equity_curve["Date"])

    # joineo los dfs por fechas

    full_df = pd.merge(equity_curve, trade_history, on="Date", how="left")

    full_df = full_df[~full_df["NetPnL"].isna()]

    # Porcentaje de ganancia

    full_df["pct"] = full_df["NetPnL"] / full_df["Equity"].shift(1)

    # Parámetros iniciales

    n_steps = len(trade_history)
    mean_return = full_df["pct"].mean()
    std_return = full_df["pct"].std()

    drawdowns_pct = []  # Lista para almacenar los drawdowns en porcentaje
    final_returns_pct = []  # Lista para almacenar los retornos finales en porcentaje
    ruin_count = 0  # Contador de simulaciones que alcanzan la ruina
    ruin_threshold = (
        initial_equity * threshold_ruin
    )  # Umbral de ruina en términos de equidad

    # Simulaciones de Montecarlo

    for _ in range(n_simulations):
        # Generar retornos aleatorios con media y desviación estándar de los históricos

        random_returns = mean_return + std_return * np.random.standard_t(15, size=n_steps)

        # Calcular la curva de equidad acumulada

        synthetic_equity_curve = initial_equity * np.cumprod(1 + random_returns)

        # Calcular drawdown y almacenarlo en porcentaje

        dd_pct = max_drawdown(synthetic_equity_curve)
        drawdowns_pct.append(dd_pct)

        # Calcular el retorno acumulado porcentual y almacenarlo

        final_return_pct = (
            synthetic_equity_curve[-1] / initial_equity - 1
        ) * 100  # Retorno final en porcentaje
        final_returns_pct.append(final_return_pct)

        # Verificar si la equidad cae por debajo del umbral de ruina en algún punto

        if np.any(synthetic_equity_curve <= ruin_threshold):
            ruin_count += 1
    # Crear un DataFrame separado para los drawdowns y los retornos acumulados en porcentaje

    df_drawdowns = pd.DataFrame({"Drawdown (%)": drawdowns_pct})
    df_final_returns_pct = pd.DataFrame({"Final Return (%)": final_returns_pct})

    # Calcular las estadísticas usando df.describe() para cada DataFrame

    if not percentiles:
        drawdown_stats = df_drawdowns.describe()
        return_stats = df_final_returns_pct.describe()
    else:
        drawdown_stats = df_drawdowns.describe(percentiles=percentiles)
        return_stats = df_final_returns_pct.describe(percentiles=percentiles)
    # Calcular el riesgo de ruina

    risk_of_ruin = ruin_count / n_simulations

    # Agregar el riesgo de ruina a las estadísticas de drawdown

    drawdown_stats.loc["Risk of Ruin"] = risk_of_ruin

    # Combinar las métricas de drawdowns y retornos porcentuales

    combined_stats = pd.concat([drawdown_stats, return_stats], axis=1)
    if return_raw_curves:
        return combined_stats, df_drawdowns, df_final_returns_pct
    
    return combined_stats

def monte_carlo_simulation_v2(
    equity_curve,
    trade_history,
    n_simulations,
    initial_equity,
    threshold_ruin,
    return_raw_curves,
    percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
):
    """
    Simulación de Monte Carlo para un sistema de trading con distribución basada en probabilidades de trades.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'NetPnL' (Profit and Loss) y 'Type' ('long' o 'short').
        equity_start (float): Valor inicial del equity.
        num_simulations (int): Número de simulaciones a realizar.
        threshold (float): Umbral para calcular el riesgo de ruina.

    Returns:
        dict: Resultados estadísticos de las simulaciones, incluyendo drawdowns y retornos.
    """
    # Filtrar trades por tipo y resultados
    trade_history.ReturnPct = trade_history.ReturnPct / 100
    
    long_trades = trade_history[trade_history['Size'] > 0]
    short_trades = trade_history[trade_history['Size'] < 0]
    
    long_winning_trades = long_trades[long_trades['NetPnL'] > 0]
    short_winning_trades = short_trades[short_trades['NetPnL'] > 0]
    long_losing_trades = long_trades[long_trades['NetPnL'] <= 0]
    short_losing_trades = short_trades[short_trades['NetPnL'] <= 0]

    # Calcular estadísticas para trades
    prob_trade = len(trade_history) / len(equity_curve)  # Probabilidad de realizar un trade
    prob_long = len(long_trades) / len(trade_history) if len(trade_history) > 0 else 0
    prob_short = len(short_trades) / len(trade_history) if len(trade_history) > 0 else 0
    prob_long_winner = len(long_winning_trades) / len(long_trades) if len(long_trades) > 0 else 0
    prob_short_winner = len(short_winning_trades) / len(short_trades) if len(short_trades) > 0 else 0
    
    long_win_mean, long_win_std = long_winning_trades['ReturnPct'].mean(), long_winning_trades['ReturnPct'].std()
    long_loss_mean, long_loss_std = long_losing_trades['ReturnPct'].mean(), long_losing_trades['ReturnPct'].std()
    short_win_mean, short_win_std = short_winning_trades['ReturnPct'].mean(), short_winning_trades['ReturnPct'].std()
    short_loss_mean, short_loss_std = short_losing_trades['ReturnPct'].mean(), short_losing_trades['ReturnPct'].std()

    # Inicializar resultados
    equity_curves = []
    drawdowns = []
    returns = []

    ruin_count = 0
    for _ in range(n_simulations):
        equity = [initial_equity]  # Curva de equity inicial

        for _ in range(len(equity_curve)):
            # Decidir si se realiza un trade
            if np.random.rand() < prob_trade:
                # Decidir si es long o short
                if np.random.rand() < prob_long:
                    # Decidir si el long es ganador o perdedor
                    if np.random.rand() < prob_long_winner:
                        trade = np.random.normal(long_win_mean, long_win_std)
                    else:
                        trade = np.random.normal(long_loss_mean, long_loss_std)
                else:
                    # Decidir si el short es ganador o perdedor
                    if np.random.rand() < prob_short_winner:
                        trade = np.random.normal(short_win_mean, short_win_std)
                    else:
                        trade = np.random.normal(short_loss_mean, short_loss_std)
            else:
                trade = 0  # No se realiza trade

            # Actualizar la curva de equity
            equity.append(equity[-1] +  equity[-1] * trade)

        # Calcular drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100 # Drawdown en porcentaje

        # Calcular retorno final
        ret = ((equity[-1] - initial_equity) / initial_equity) * 100  # Retorno en porcentaje
        
        if np.any(np.array(equity) <= initial_equity * threshold_ruin):
            ruin_count += 1

        # Guardar resultados
        equity_curves.append(equity)
        drawdowns.append(dd.min())  # Máximo drawdown
        returns.append(ret)

    df_drawdowns = pd.DataFrame({"Drawdown (%)": drawdowns})
    df_final_returns_pct = pd.DataFrame({"Final Return (%)": returns})

    # Calcular las estadísticas usando df.describe() para cada DataFrame

    if not percentiles:
        drawdown_stats = df_drawdowns.describe()
        return_stats = df_final_returns_pct.describe()
    else:
        drawdown_stats = df_drawdowns.describe(percentiles=percentiles)
        return_stats = df_final_returns_pct.describe(percentiles=percentiles)

    # Calcular el riesgo de ruina

    risk_of_ruin = (ruin_count / n_simulations) * 100
    drawdown_stats.loc["Risk of Ruin"] = risk_of_ruin

    # Combinar las métricas de drawdowns y retornos porcentuales

    combined_stats = pd.concat([drawdown_stats, return_stats], axis=1)
    if return_raw_curves:
        return combined_stats, df_drawdowns, df_final_returns_pct
    
    return combined_stats