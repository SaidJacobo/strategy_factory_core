import os
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import yaml
from app.backbone.database.db_service import DbService
from app.backbone.entities.bot_performance import BotPerformance
from app.backbone.entities.bot_trade_performance import BotTradePerformance
from app.backbone.entities.luck_test import LuckTest
from app.backbone.entities.metric_wharehouse import MetricWharehouse
from app.backbone.entities.montecarlo_test import MontecarloTest
from app.backbone.entities.random_test import RandomTest
from app.backbone.entities.trade import Trade
from app.backbone.services.backtest_service import BacktestService
from app.backbone.services.config_service import ConfigService
from app.backbone.services.operation_result import OperationResult
from app.backbone.services.utils import _performance_from_df_to_obj, calculate_sharpe_ratio, get_trade_df_from_db
from app.backbone.utils.get_data import get_data
from app.backbone.utils.general_purpose import load_function
from app.backbone.utils.montecarlo_utils import max_drawdown, monte_carlo_simulation_v2
from app.backbone.utils.wfo_utils import run_strategy_and_get_performances
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class TestService:
    """
    Provides a collection of statistical and robustness tests for evaluating trading bot performance.

    This service encapsulates methods that assess the quality, reliability, and randomness of 
    a strategy's results by applying various statistical analyses such as:
    - t-tests for Sharpe ratio significance
    - correlation tests between bot returns and the underlying asset
    - randomization tests against synthetic strategies
    - luck-based filtering of extreme trades
    - Monte Carlo simulations to evaluate ruin probabilities

    Attributes:
        db_service (DbService): Handles database interactions and persistence.
        backtest_service (BacktestService): Retrieves historical bot performance and trade data.
        config_service (ConfigService): Provides access to application-wide configuration values (e.g., risk-free rate).
    
    Notes:
        - This class is intended to be used as a backend utility for validating trading bot robustness.
        - Each test method typically returns an `OperationResult` and may generate visualizations or database entries.
    """
    def __init__(self):
        self.db_service = DbService()
        self.backtest_service = BacktestService()
        self.config_service = ConfigService()
        
    def run_montecarlo_test(self, bot_performance_id:int, n_simulations:int, threshold_ruin:float) -> OperationResult:
        """
        Performs a Monte Carlo simulation to assess the probabilistic risk profile of a bot's equity curve.

        This test runs multiple randomized simulations of the bot’s historical trade sequence to estimate 
        the potential distribution of future outcomes. It focuses on downside risk by evaluating the 
        likelihood of reaching a "ruin" threshold and analyzing performance percentiles.

        Steps performed:
        - Loads the bot's historical trade and equity data.
        - Runs `n_simulations` Monte Carlo paths using the historical trade returns.
        - Calculates key percentile statistics (e.g., 5th, 10th, median, 90th, 95th) over all simulations.
        - Evaluates whether the equity would fall below the specified `threshold_ruin`.
        - Stores the test metadata and resulting metrics in the database.

        Parameters:
            bot_performance_id (int): The ID of the bot performance record to analyze.
            n_simulations (int): Number of Monte Carlo simulations to run.
            threshold_ruin (float): Capital threshold considered as "ruin" (e.g., 0.3 = 30% of initial capital).

        Returns:
            OperationResult: Indicates success and contains a list of `MetricWharehouse` entries 
            with computed statistics.

        Side effects:
            - Saves a `MontecarloTest` entity in the database linked to the given performance.
            - Stores all percentile results and associated metadata in the `MetricWharehouse` table.

        Notes:
            - This test provides a probabilistic view of strategy robustness under randomness and path dependency.
            - Useful for risk management, especially when assessing worst-case scenarios or capital preservation.
            - The simulation respects the statistical properties of the original trade distribution (returns and durations).
        """
        performance = self.backtest_service.get_bot_performance_by_id(bot_performance_id=bot_performance_id)
        trades_history = get_trade_df_from_db(performance.TradeHistory, performance_id=performance.Id)

        mc = monte_carlo_simulation_v2(
            equity_curve=trades_history.Equity,
            trade_history=trades_history,
            n_simulations=n_simulations,
            initial_equity=performance.InitialCash,
            threshold_ruin=threshold_ruin,
            return_raw_curves=False,
            percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        )

        mc = mc.round(3).reset_index().rename(
            columns={'index':'metric'}
        )
        
        mc_long = mc.melt(id_vars=['metric'], var_name='ColumnName', value_name='Value').fillna(0)
        
        montecarlo_test = MontecarloTest(
            BotPerformanceId=performance.Id,
            Simulations=n_simulations,
            ThresholdRuin=threshold_ruin,
        )
        
        rows = [
            MetricWharehouse(
                Method='Montecarlo', 
                Metric=row['metric'], 
                ColumnName=row['ColumnName'], 
                Value=row['Value'],
                MontecarloTest=montecarlo_test
            )
            
            for _, row in mc_long.iterrows()
        ]
        
        with self.db_service.get_database() as db:
            self.db_service.create(db, montecarlo_test)
            self.db_service.create_all(db, rows)
        
        return OperationResult(ok=True, message=None, item=rows)
               
    def run_luck_test(self, bot_performance_id, trades_percent_to_remove) -> OperationResult:
        """
        Performs a "luck test" to evaluate the robustness of a bot's performance by removing extreme trades.

        This method simulates the impact of luck by discarding a percentage of the bot's best and worst trades,
        then recalculating performance metrics based on the remaining data. The goal is to estimate how much
        of the bot's performance depends on a few outliers.

        Steps performed:
        - Loads the bot's trade history and determines how many trades to remove based on the given percentage.
        - Identifies the top `X%` best and worst trades by return percentage.
        - Filters out these trades and rebuilds the equity curve from the remaining ones.
        - Calculates new performance metrics: return, drawdown, return/drawdown ratio, winrate, and a custom metric.
        - Fits a linear regression to the equity curve to compute a stability ratio.
        - Creates a new `BotPerformance` object based on the filtered trades.
        - Stores all results, including which trades were marked as "lucky" (best or worst), in the database.
        - Triggers the generation of a Plotly chart to visualize the filtered performance.

        Parameters:
            bot_performance_id (int): The ID of the bot performance record to analyze.
            trades_percent_to_remove (float): Percentage of top and bottom trades (by return) to remove for the test.

        Returns:
            OperationResult: An object indicating success and containing the resulting `LuckTest` record.

        Side effects:
            - Updates the trade records in the database to flag trades as top-best or top-worst.
            - Saves a new `LuckTest` and `BotPerformance` entry reflecting the filtered results.
            - Generates and stores a visualization of the adjusted performance.

        Notes:
            - This test helps detect whether a bot's profitability is overly dependent on a few outlier trades.
            - The stability ratio is derived from the R² of a linear regression on the filtered equity curve.
            - The custom metric is a weighted risk-adjusted return penalized by drawdown and enhanced by stability.
        """
        performance = self.backtest_service.get_bot_performance_by_id(bot_performance_id=bot_performance_id)
        trades = get_trade_df_from_db(performance.TradeHistory, performance_id=performance.Id)
        trades_to_remove = round((trades_percent_to_remove/100) * trades.shape[0])
        
        top_best_trades = trades.sort_values(by='ReturnPct', ascending=False).head(trades_to_remove)
        top_worst_trades = trades.sort_values(by='ReturnPct', ascending=False).tail(trades_to_remove)
        
        filtered_trades = trades[
            (~trades['Id'].isin(top_best_trades.Id))
            & (~trades['Id'].isin(top_worst_trades.Id))
            & (~trades['ReturnPct'].isna())
        ].sort_values(by='ExitTime')

        filtered_trades.ReturnPct = filtered_trades.ReturnPct / 100

        filtered_trades['Equity'] = 0
        filtered_trades['Equity'] = (performance.InitialCash * (1 + filtered_trades.ReturnPct).cumprod()).round(3)
        
        dd = np.abs(max_drawdown(filtered_trades['Equity'])).round(3)
        ret = ((filtered_trades.iloc[-1]['Equity'] - filtered_trades.iloc[0]['Equity']) / filtered_trades.iloc[0]['Equity']) * 100
        ret = round(ret, 3)
        
        ret_dd = (ret / dd).round(3)
        
        x = np.arange(filtered_trades.shape[0]).reshape(-1, 1)
        reg = LinearRegression().fit(x, filtered_trades['Equity'])
        stability_ratio = round(reg.score(x, filtered_trades['Equity']), 3)

        custom_metric = ((ret / (1 + dd)) * np.log(1 + filtered_trades.shape[0])).round(3) * stability_ratio
        
        new_winrate = round(
            (filtered_trades[filtered_trades['NetPnL']>0]['Id'].size / filtered_trades['Id'].size) * 100, 3
        )
        
        luck_test_performance = BotPerformance(**{
            'DateFrom': performance.DateFrom,
            'DateTo': performance.DateTo,
            'BotId': None,
            'StabilityRatio': stability_ratio,
            'Trades': filtered_trades['Id'].size,
            'Return': ret,
            'Drawdown': dd,
            'RreturnDd': ret_dd,
            'WinRate': new_winrate,
            'Duration': performance.Duration,
            'StabilityWeightedRar': custom_metric,
            'Method': 'luck_test',
            'InitialCash': performance.InitialCash,
            'ExposureTime': performance.ExposureTime,
            'KellyCriterion': performance.KellyCriterion,
            'WinratePValue': performance.WinratePValue
        })
        
        luck_test = LuckTest(**{
            'BotPerformanceId': performance.Id,
            'TradesPercentToRemove': trades_percent_to_remove,
            'LuckTestPerformance': luck_test_performance
        })
        
        top_best_trades_id = top_best_trades['Id'].values
        top_worst_trades_id = top_worst_trades['Id'].values
        
        with self.db_service.get_database() as db:
            
            for trade in performance.TradeHistory:
                if trade.Id in top_best_trades_id:
                    trade.TopBest = True
                    
                if trade.Id in top_worst_trades_id:
                    trade.TopWorst = True
                
                _ = self.db_service.update(db, Trade, trade)
            
            luck_test_db = self.db_service.create(db, luck_test)
            _ = self.db_service.create(db, luck_test_performance)
        
        self._create_luck_test_plot(bot_performance_id=bot_performance_id)
    
        return OperationResult(ok=True, message=None, item=luck_test_db)

    def get_luck_test_equity_curve(self, bot_performance_id, remove_only_good_luck=False) -> OperationResult:
        """
        Generates a filtered equity curve by removing lucky trades from a bot's historical performance.

        This method rebuilds the bot's equity curve after excluding trades flagged as either top-performing 
        ("TopBest"), worst-performing ("TopWorst"), or both, depending on the configuration.

        Parameters:
            bot_performance_id (int): The ID of the bot performance record to analyze.
            remove_only_good_luck (bool): If True, only the top-performing trades ("TopBest") are removed.
                                        If False, both "TopBest" and "TopWorst" trades are removed.

        Returns:
            OperationResult: Contains the filtered equity curve as a DataFrame with `ExitTime` and `Equity`.

        Notes:
            - This function is typically used to assess the strategy's robustness by simulating less favorable luck.
            - Equity is recomputed using cumulative product of filtered returns.
        """
        performance = self.backtest_service.get_bot_performance_by_id(bot_performance_id=bot_performance_id)
        trades = get_trade_df_from_db(performance.TradeHistory, performance_id=performance.Id)
        
        if remove_only_good_luck:
            filtered_trades = trades[(trades['TopBest'].isna())].sort_values(by='ExitTime')
        
        else:
            filtered_trades = trades[(trades['TopBest'].isna()) & (trades['TopWorst'].isna())].sort_values(by='ExitTime')
        
        filtered_trades['Equity'] = 0
        filtered_trades.ReturnPct = filtered_trades.ReturnPct / 100
        filtered_trades['Equity'] = (performance.InitialCash * (1 + filtered_trades.ReturnPct).cumprod()).round(3)
        equity = filtered_trades[['ExitTime','Equity']]
        
        return OperationResult(ok=True, message=None, item=equity)

    def _create_luck_test_plot(self, bot_performance_id) -> OperationResult:
        """
        Generates and saves a Plotly chart comparing the original and "luck test" equity curves.

        This function plots three equity curves:
        - The original equity curve with all trades.
        - The equity curve after removing both top and bottom trades (luck-neutral).
        - The equity curve after removing only top-performing trades (bad-luck only).

        Steps performed:
        - Retrieves original and filtered trade histories.
        - Computes equity for each curve.
        - Plots them using Plotly and exports the figure as a JSON file.

        Parameters:
            bot_performance_id (int): The ID of the bot performance record to visualize.

        Returns:
            OperationResult: Indicates success or failure. The chart is saved as a JSON file for rendering.

        Side effects:
            - Saves the chart to `./app/templates/static/luck_test_plots` with a filename
            based on the bot's name and performance date range.

        Notes:
            - Helps visualize how the strategy performs with or without "lucky" trades.
            - This is a private helper method not intended to be used directly.
        """
        bot_performance = self.backtest_service.get_bot_performance_by_id(bot_performance_id=bot_performance_id)
        bot_performance.TradeHistory = sorted(bot_performance.TradeHistory, key=lambda trade: trade.ExitTime)

        # Equity plot
        dates = [trade.ExitTime for trade in bot_performance.TradeHistory]
        equity = [trade.Equity for trade in bot_performance.TradeHistory]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity,
                            mode='lines',
                            name='Equity'))

        print('Calculando curva de luck test')
        result = self.get_luck_test_equity_curve(bot_performance_id)

        luck_test_equity_curve = result.item
        print(luck_test_equity_curve)
        
        print('Calculando curva de luck test (BL)')
        result = self.get_luck_test_equity_curve(bot_performance_id, remove_only_good_luck=True)
        if not result.ok:
            return result
        
        luck_test_remove_only_good = result.item
    
        fig.add_trace(go.Scatter(x=luck_test_equity_curve.ExitTime, y=luck_test_equity_curve.Equity,
                            mode='lines',
                            name=f'Luck test'))
        
        fig.add_trace(go.Scatter(x=luck_test_remove_only_good.ExitTime, y=luck_test_remove_only_good.Equity,
                            mode='lines',
                            name=f'Luck test (BL)'))

        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Equity'
        )   
        
        str_date_from = str(bot_performance.DateFrom).replace('-','')
        str_date_to = str(bot_performance.DateTo).replace('-','')
        file_name=f'{bot_performance.Bot.Name}_{str_date_from}_{str_date_to}.html'
        
        print('Guardando grafico')
        
        plot_path = './app/templates/static/luck_test_plots'
        
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        json_content = fig.to_json()

        with open(os.path.join(plot_path, file_name), 'w') as f:
            f.write(json_content)
            
    def run_random_test(self, bot_performance_id, n_iterations) -> OperationResult:
        """
        Performs a Monte Carlo randomization test to evaluate the statistical significance of a bot's performance.

        This method compares the real bot's performance metrics to those obtained from multiple randomized simulations
        that mimic the bot's trade frequency, duration, and direction probabilities. The goal is to assess whether 
        the bot’s performance could have occurred by chance.

        Steps performed:
        - Loads bot performance data, price history, and equity curve.
        - Extracts empirical trading behavior (probabilities, duration statistics).
        - Runs a set of randomized backtests using a custom "RandomTrader" strategy to simulate noise-based trades.
        - Bootstraps real and random returns `n_iterations` times and computes key performance metrics.
        - For each metric (return, drawdown, return/DD, winrate), calculates:
            - Mean and standard deviation differences
            - Z-scores
            - One-sided p-values (probability that the real strategy performs worse or equal to the random one).
        - Stores the statistical results in the database.

        Parameters:
            bot_performance_id (int): The ID of the bot performance record to analyze.
            n_iterations (int): Number of bootstrap iterations to compare real vs. random performance.

        Returns:
            OperationResult: An object indicating success. Results are saved to the database.

        Side effects:
            - Runs multiple randomized backtests using the bot's historical market data.
            - Saves statistical results in a `RandomTest` table (via `self.db_service`).

        Notes:
            - This test evaluates whether the strategy adds value beyond what could be expected by random chance.
            - Performance metrics compared: total return, max drawdown, return-to-drawdown ratio, and winrate.
            - Assumes the presence of a random strategy class at `'app.backbone.strategies.random_trader.RandomTrader'`.
        """
        bot_performance = self.backtest_service.get_bot_performance_by_id(bot_performance_id=bot_performance_id)
        ticker = bot_performance.Bot.Ticker
        timeframe = bot_performance.Bot.Timeframe
        
        with open("./app/configs/leverages.yml", "r") as file_name:
            leverages = yaml.safe_load(file_name)
            
        leverage = leverages[ticker.Name]
        
        strategy_path = 'app.backbone.strategies.random_trader.RandomTrader'
        
        strategy_func = load_function(strategy_path)
        
        trade_history = get_trade_df_from_db(
            bot_performance.TradeHistory, 
            performance_id=bot_performance.Id
        )
        
        long_trades = trade_history[trade_history['Size'] > 0]
        short_trades = trade_history[trade_history['Size'] < 0]
        
        date_from = pd.Timestamp(bot_performance.DateFrom, tz="UTC")
        date_to = pd.Timestamp(bot_performance.DateTo, tz="UTC")

        prices = get_data(
            ticker.Name, 
            timeframe.MetaTraderNumber, 
            date_from, 
            date_to
        )
        
        prices.index = pd.to_datetime(prices.index)
        
        prob_trade = len(trade_history) / len(prices)  # Probabilidad de realizar un trade
        prob_long = len(long_trades) / len(trade_history) if len(trade_history) > 0 else 0
        prob_short = len(short_trades) / len(trade_history) if len(trade_history) > 0 else 0
        
        trade_history["Duration"] = pd.to_timedelta(trade_history["Duration"])
        trade_history["Bars"] = trade_history["ExitBar"] - trade_history["EntryBar"]

        avg_trade_duration = trade_history.Bars.mean()
        std_trade_duration = trade_history.Bars.std()

        params = {
            'prob_trade': prob_trade,
            'prob_long': prob_long,
            'prob_short': prob_short,
            'avg_trade_duration': avg_trade_duration,
            'std_trade_duration': std_trade_duration,
        }
        
        risk_free_rate = float(self.config_service.get_by_name('RiskFreeRate').Value)


        all_random_trades = pd.DataFrame()
        for _ in range(10):
            _, _, stats = run_strategy_and_get_performances(
                strategy=strategy_func,
                ticker=ticker,
                timeframe=timeframe,
                risk=bot_performance.Bot.Risk,
                prices=prices,
                initial_cash=bot_performance.InitialCash,
                risk_free_rate=risk_free_rate,
                margin=1 / leverage, 
                opt_params=params
            )

            all_random_trades = pd.concat([
                all_random_trades,
                stats._trades
            ])

        np.random.seed(42)

        n_real = len(trade_history)

        returns_real = trade_history.ReturnPct / 100
        returns_rand = all_random_trades.ReturnPct / 100

        metrics = {
            'return': lambda eq: ((eq[-1] - eq[0]) / eq[0]) * 100,
            'dd': lambda eq: max_drawdown(eq).round(3),
            'return_dd': lambda eq: ((eq[-1] - eq[0]) / eq[0]) * 100 / abs(max_drawdown(eq)),
            'winrate': lambda r: np.mean(r > 0)
        }

        real_results = {k: [] for k in metrics}
        rand_results = {k: [] for k in metrics}

        for _ in range(n_iterations):
            sample_real = np.random.choice(returns_real, size=n_real, replace=True)
            sample_rand = np.random.choice(returns_rand, size=n_real, replace=True)

            equity_real = bot_performance.InitialCash * (1 + sample_real).cumprod()
            equity_rand = bot_performance.InitialCash * (1 + sample_rand).cumprod()

            for name, func in metrics.items():
                if name == 'winrate':
                    real_results[name].append(func(sample_real))
                    rand_results[name].append(func(sample_rand))
                else:
                    real_results[name].append(func(equity_real))
                    rand_results[name].append(func(equity_rand))

        # Evaluación estadística
        p_values = {}
        z_scores = {}
        mean_diffs = {}
        std_diffs = {}

        for name in metrics:
            diffs = np.array(real_results[name]) - np.array(rand_results[name])
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            z_scores[name] = mean_diff / std_diff
            p_values[name] = np.mean(diffs <= 0)  # test unidireccional, proporción de veces que la estrategia real fue igual o peor que la random. 
            mean_diffs[name] = mean_diff
            std_diffs[name] = std_diff

        
        with self.db_service.get_database() as db:
            # Primero, guardar random_test_performance_for_db
            
            # Ahora que tenemos el ID, podemos asignarlo a random_test
            random_test = RandomTest(
                Iterations=n_iterations,
                BotPerformanceId=bot_performance.Id,
                ReturnDdMeanDiff=round(mean_diffs['return_dd'], 3),
                ReturnDdStdDiff=round(std_diffs['return_dd'], 3),
                ReturnDdPValue=round(p_values['return_dd'], 5),
                ReturnDdZScore=round(z_scores['return_dd'], 3),

                ReturnMeanDiff=round(mean_diffs['return'], 3),
                ReturnStdDiff=round(std_diffs['return'], 3),
                ReturnPValue=round(p_values['return'], 5),
                ReturnZScore=round(z_scores['return'], 3),

                DrawdownMeanDiff=round(mean_diffs['dd'], 3),
                DrawdownStdDiff=round(std_diffs['dd'], 3),
                DrawdownPValue=round(p_values['dd'], 5),
                DrawdownZScore=round(z_scores['dd'], 3),

                WinrateMeanDiff=round(mean_diffs['winrate'], 3),
                WinrateStdDiff=round(std_diffs['winrate'], 3),
                WinratePValue=round(p_values['winrate'], 5),
                WinrateZScore=round(z_scores['winrate'], 3),
            )
            
            # Guardar random_test ahora con la relación correcta
            self.db_service.create(db, random_test)
            
        return OperationResult(ok=True, message=None, item=None)

    def run_correlation_test(self, bot_performance_id: int) -> OperationResult:
        """
        Performs a correlation analysis between a bot's monthly returns and the underlying asset's price variation.

        This method calculates the monthly percentage changes in the bot's equity curve and compares them to
        the monthly price changes of the underlying instrument. It evaluates both the Pearson correlation 
        coefficient and the coefficient of determination (R²) via linear regression.

        Steps performed:
        - Retrieves the bot's equity curve and historical price data for the configured ticker and timeframe.
        - Aggregates both series to monthly frequency and computes percentage variations.
        - Aligns and fills missing months to ensure matching time indices.
        - Fits a linear regression model between asset price changes and bot returns.
        - Computes Pearson's correlation coefficient and R² score.
        - Generates a Plotly scatter plot with a fitted regression line.
        - Annotates the plot with the correlation and determination values.
        - Saves the plot as a JSON file for rendering.

        Parameters:
            bot_performance_id (int): The ID of the bot performance record to analyze.

        Returns:
            OperationResult: An object indicating success and containing a DataFrame with:
                - `correlation`: Pearson correlation coefficient (r)
                - `determination`: Coefficient of determination (R²)

        Side effects:
            Saves a Plotly chart as an HTML-ready JSON file in the `./app/templates/static/correlation_plots` directory.
            The filename includes the bot's name and performance date range.

        Notes:
            - This analysis helps identify the degree to which the bot's performance is correlated with the underlying asset.
            - Monthly data is used to smooth out noise and capture general trends.
        """

        bot_performance = self.backtest_service.get_bot_performance_by_id(bot_performance_id=bot_performance_id)
        
        trade_history = get_trade_df_from_db(
            bot_performance.TradeHistory, 
            performance_id=bot_performance.Id
        )
        
        prices = get_data(
            bot_performance.Bot.Ticker.Name, 
            bot_performance.Bot.Timeframe.MetaTraderNumber, 
            pd.Timestamp(bot_performance.DateFrom, tz="UTC"), 
            pd.Timestamp(bot_performance.DateTo, tz="UTC")
        )

        # Transformar el índice al formato mensual
        trade_history = trade_history.reset_index()
        equity = trade_history[['ExitTime', 'Equity']]
                
        equity['month'] = pd.to_datetime(equity['ExitTime']).dt.to_period('M')
        equity = equity.groupby(by='month').agg({'Equity': 'last'})
        equity['perc_diff'] = (equity['Equity'] - equity['Equity'].shift(1)) / equity['Equity'].shift(1)
        equity.fillna(0, inplace=True)

        # Crear un rango completo de meses con PeriodIndex
        full_index = pd.period_range(start=equity.index.min(), end=equity.index.max(), freq='M')

        # Reindexar usando el rango completo de PeriodIndex
        equity = equity.reindex(full_index)
        equity = equity.ffill()

        prices['month'] = pd.to_datetime(prices.index)
        prices['month'] = prices['month'].dt.to_period('M')
        prices = prices.groupby(by='month').agg({'Close':'last'})
        prices['perc_diff'] = (prices['Close'] - prices['Close'].shift(1)) / prices['Close'].shift(1)
        prices.fillna(0, inplace=True)

        prices = prices[prices.index.isin(equity.index)]

        x = np.array(prices['perc_diff']).reshape(-1, 1)
        y = equity['perc_diff']

        # Ajustar el modelo de regresión lineal
        reg = LinearRegression().fit(x, y)
        determination = reg.score(x, y)
        correlation = np.corrcoef(prices['perc_diff'], equity['perc_diff'])[0, 1]

        # Predicciones para la recta
        x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)  # Rango de X para la recta
        y_pred = reg.predict(x_range)  # Valores predichos de Y

        result = pd.DataFrame({
            'correlation': [correlation],
            'determination': [determination],
        }).round(3)

        # Crear el gráfico
        fig = px.scatter(
            x=prices['perc_diff'], y=equity['perc_diff'],
        )

        # Agregar la recta de regresión
        fig.add_scatter(x=x_range.flatten(), y=y_pred, mode='lines', name='Regresión Lineal')

        # Personalización
        fig.update_layout(
            xaxis_title='Monthly Price Variation',
            yaxis_title='Monthly Returns'
        )

        # Agregar anotación con los valores R² y Pearson
        fig.add_annotation(
            x=0.95,  # Posición en el gráfico (en unidades de fracción del eje)
            y=0.95,
            xref='paper', yref='paper',
            text=f"<b>r = {correlation:.3f}<br>R² = {determination:.3f}</b>",
            showarrow=False,
            font=dict(size=16, color="black"),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )

        str_date_from = str(bot_performance.DateFrom).replace('-','')
        str_date_to = str(bot_performance.DateTo).replace('-','')
        file_name=f'{bot_performance.Bot.Name}_{str_date_from}_{str_date_to}.html'
        
        plot_path='./app/templates/static/correlation_plots'
        
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        json_content = fig.to_json()

        with open(os.path.join(plot_path, file_name), 'w') as f:
            f.write(json_content)
        
        return OperationResult(ok=True, message=False, item=result)
    
    def run_t_test(self, bot_performance_id: int):
        """
        Performs a rolling Sharpe Ratio analysis with confidence intervals for a given bot's performance.

        This method calculates the monthly returns of the bot based on its equity curve and evaluates
        the statistical stability of its risk-adjusted performance over time.

        Steps performed:
        - Extracts the bot's equity history and computes monthly returns.
        - Tests for normality using the Shapiro-Wilk test.
        - Computes the cumulative Sharpe Ratio month-by-month.
        - Builds 95% confidence intervals using either:
            - Bootstrapping (if returns are not normally distributed), or
            - Theoretical t-distribution (if returns are normal).
        - Renders an interactive Plotly chart with the Sharpe Ratio curve and confidence bands.
        - Annotates the Shapiro-Wilk p-value and saves the chart as a JSON file.

        Parameters:
            bot_performance_id (int): The ID of the bot performance record to analyze.

        Returns:
            OperationResult: A success indicator and optional message or payload.
        
        Side effects:
            Saves a Plotly chart as an HTML-ready JSON file in the `./app/templates/static/t_test_plots` directory.
            The filename includes the bot's name and performance date range.

        Notes:
            - The risk-free rate is retrieved from the configuration under the key "RiskFreeRate".
            - The Sharpe Ratio is annualized assuming 12 periods (monthly data).
            - The analysis starts from the second available month (since returns require two data points).
        """
        risk_free_rate = float(self.config_service.get_by_name('RiskFreeRate').Value)
        
        def bootstrap_sharpe_ci_cumulative(returns, n_bootstrap=1000, confidence=0.95):
            ci_lower = []
            ci_upper = []
            sharpe_cumulative = []

            for i in range(1, len(returns)):
                current_returns = returns[:i+1]
                n = len(current_returns)
                sr_bootstrap = []

                for _ in range(n_bootstrap):
                    sample = np.random.choice(current_returns, size=n, replace=True)
                    sr = calculate_sharpe_ratio(sample, risk_free_rate=risk_free_rate, trading_periods=12)
                    sr_bootstrap.append(sr)

                lower = np.percentile(sr_bootstrap, (1 - confidence) / 2 * 100)
                upper = np.percentile(sr_bootstrap, (1 + confidence) / 2 * 100)
                point_estimate = calculate_sharpe_ratio(current_returns, risk_free_rate=risk_free_rate, trading_periods=12)

                ci_lower.append(lower)
                ci_upper.append(upper)
                sharpe_cumulative.append(point_estimate)

            return sharpe_cumulative, ci_lower, ci_upper

        bot_performance = self.backtest_service.get_bot_performance_by_id(bot_performance_id=bot_performance_id)
        
        trade_history = get_trade_df_from_db(
            bot_performance.TradeHistory, 
            performance_id=bot_performance.Id
        )
 
        # Transformar el índice al formato mensual
        trade_history = trade_history.reset_index()
        equity = trade_history[['ExitTime', 'Equity']]

        # Asegurar formato de fecha y agrupar por mes
        equity['ExitTime'] = pd.to_datetime(equity['ExitTime'])
        equity['month'] = equity['ExitTime'].dt.to_period('M')
        equity = equity.groupby(by='month').agg({'Equity': 'last'})

        # Calcular retornos mensuales
        returns = equity['Equity'].pct_change().dropna().values
    
        shapiro_test = stats.shapiro(returns)
        is_normal = shapiro_test.pvalue > 0.05

        confidence = 0.95
        if not is_normal:
            sharpe_cumulative, ci_lower, ci_upper = bootstrap_sharpe_ci_cumulative(returns, confidence=confidence)
            
        else:
            sharpe_cumulative = [calculate_sharpe_ratio(returns[:i+1], risk_free_rate=risk_free_rate, trading_periods=12) for i in range(1, len(returns))]
            ci_lower = []
            ci_upper = []
            for i in range(1, len(returns)):
                n = i+1
                current_returns = returns[:n]
                sr = calculate_sharpe_ratio(current_returns, risk_free_rate=risk_free_rate, trading_periods=12)
                se = np.sqrt((1 + 0.5*sr**2) / n)
                t = stats.t.ppf((1 + confidence)/2, df=n-1)
                ci_lower.append(sr - t*se)
                ci_upper.append(sr + t*se)
        
        x_axis_labels = equity.index[1:].astype(str)
        
        # Create traces
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis_labels, y=sharpe_cumulative,
                            mode='lines+markers', 
                            name='Sharpe Ratio', 
                            line=dict(color='blue')))
        
        fig.add_trace(go.Scatter(x=x_axis_labels, 
                                 y=ci_upper,
                                 mode='lines', 
                                 name='Upper limit 95%', 
                                 line=dict(color='green', dash='dash')))
        
        fig.add_trace(go.Scatter(x=x_axis_labels, 
                                 y=ci_lower,mode='lines', 
                                 name='Lower Limit 95%', 
                                 line=dict(color='red', dash='dash')))
        
        fig.add_trace(go.Scatter(x=x_axis_labels, 
                                 y=np.zeros(shape=(len(sharpe_cumulative))),
                                 mode='lines', 
                                 name='Zero', 
                                 line=dict(color='black')))
        
        fig.add_annotation(
            x=0.95,  # Posición en el gráfico (en unidades de fracción del eje)
            y=0.95,
            xref='paper', yref='paper',
            text=f"Shapiro-Wilk p-value: {shapiro_test.pvalue:.4f}",
            showarrow=False,
            font=dict(size=12, color="red" if not is_normal else "green"),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )

        str_date_from = str(bot_performance.DateFrom).replace('-','')
        str_date_to = str(bot_performance.DateTo).replace('-','')
        file_name=f'{bot_performance.Bot.Name}_{str_date_from}_{str_date_to}.html'
        
        plot_path='./app/templates/static/t_test_plots'
        
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        json_content = fig.to_json()

        with open(os.path.join(plot_path, file_name), 'w') as f:
            f.write(json_content)

        return OperationResult(ok=True, message=False, item=None)
