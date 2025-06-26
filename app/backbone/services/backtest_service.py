import asyncio
from datetime import date
import itertools
import json
import os
from typing import AsyncGenerator, List
import yaml
from app.backbone.database.db_service import DbService
from app.backbone.entities.bot import Bot
from app.backbone.entities.bot_performance import BotPerformance
from app.backbone.entities.bot_trade_performance import BotTradePerformance
from app.backbone.entities.luck_test import LuckTest
from app.backbone.entities.metric_wharehouse import MetricWharehouse
from app.backbone.entities.montecarlo_test import MontecarloTest
from app.backbone.entities.random_test import RandomTest
from app.backbone.entities.strategy import Strategy
from app.backbone.entities.ticker import Ticker
from app.backbone.entities.timeframe import Timeframe
from app.backbone.entities.trade import Trade
from app.backbone.services.config_service import ConfigService
from app.backbone.services.operation_result import OperationResult
from app.backbone.services.bot_service import BotService
from app.backbone.services.utils import _performance_from_df_to_obj, trades_from_df_to_obj
from app.backbone.utils.get_data import get_data
from app.backbone.utils.general_purpose import load_function, profile_function, streaming_endpoint
from app.backbone.utils.wfo_utils import run_strategy_and_get_performances
import pandas as pd
from sqlalchemy.orm import joinedload
from sqlalchemy import delete, func, desc, select
from sqlalchemy.orm import aliased


async def log_message(queue: asyncio.Queue, ticker, timeframe, status, message, error=""):
    """Función para agregar logs a la cola sin detener el proceso."""
    await queue.put(json.dumps({
        "ticker": ticker,
        "timeframe": timeframe,
        "status": status,
        "message": message,
        "error": error
}))

class BacktestService:
    """
    Handles the orchestration, execution, persistence, and retrieval of backtests and related performance metrics.

    This service acts as the central component for running backtests on trading strategies, saving their results,
    and performing various queries on historical performance data. It supports streaming logs, asynchronous execution,
    and interaction with the database and configuration layers.

    Main features include:
    - Running and saving individual or batch backtests across strategy/ticker/timeframe/risk combinations.
    - Loading strategies dynamically and executing them with proper capital, leverage, and configuration.
    - Persisting detailed performance metrics and trade history in the database.
    - Performing deletions and cleanup of historical test data (including files and DB records).
    - Providing filtering, querying, and tagging of backtests (e.g. by performance, robustness, favorites).
    - Identifying robust strategies via custom metrics and filters (e.g. return/drawdown ratio > 1).
    - Managing test dependencies such as Monte Carlo, Luck Test, and Random Test results.

    Attributes:
        db_service (DbService): Handles database connections and CRUD operations.
        bot_service (BotService): Manages bot records used during backtesting.
        config_service (ConfigService): Provides access to configuration values (e.g. risk-free rate).

    Notes:
        - Backtests can be persisted with reports or run temporarily.
        - Streaming logs (via queues) allow real-time feedback when executing long-running backtests.
        - This class assumes a full application context with all entities (Bot, BotPerformance, Trade, etc.) properly defined.
    """
    def __init__(self):
        self.db_service = DbService()
        self.bot_service = BotService()
        self.config_service = ConfigService()
    
    async def async_iterator(self, iterable):
        # Si iterable es una lista, la envolvemos en una iteración asíncrona
        for item in iterable:
            yield item
            await asyncio.sleep(0)

    @streaming_endpoint
    async def run_backtest(
        self,
        initial_cash: float,
        strategy: Strategy,
        ticker: Ticker,
        timeframe: Timeframe,
        date_from: pd.Timestamp,
        date_to: pd.Timestamp,
        method: str,
        risk: float,
        save_bt_plot: str,
        queue: asyncio.Queue,
        ) -> AsyncGenerator[str, None]:
        """
        Executes an asynchronous backtest for a given trading strategy and returns performance metrics, 
        trade details, and statistics via a streaming endpoint.

        This function loads the specified strategy, fetches historical price data, applies leverage rules,
        and runs a backtest simulation. Results are streamed in real-time through an asyncio queue, 
        and optional plots/reports are generated.

        Steps performed:
        - Loads the trading strategy dynamically from a module path.
        - Retrieves leverage rules from a YAML config file.
        - Fetches and prepares historical price data for the given ticker/timeframe.
        - Computes margin requirements based on leverage.
        - Executes the backtest using the strategy's logic and calculates performance metrics.
        - Generates and saves backtest plots (temporary or persistent) if requested.
        - Streams progress updates, logs, and final results through the provided queue.

        Parameters:
        - initial_cash (float): Starting capital for the backtest simulation.
        - strategy (Strategy): Strategy object containing the module path and name.
        - ticker (Ticker): Financial instrument to backtest on (e.g., currency pair, stock).
        - timeframe (Timeframe): Timeframe for price data (e.g., '1H', '4H').
        - date_from (pd.Timestamp): Start date for historical data.
        - date_to (pd.Timestamp): End date for historical data.
        - method (str): Reserved for future backtest variations (unused in current implementation).
        - risk (float): Risk percentage per trade (e.g., 0.01 for 1% risk).
        - save_bt_plot (str): Plot saving mode: 
            - 'temp': Saves in a temporary directory (auto-cleaned later).
            - 'persist': Saves in the main backtest_plots directory.
            - Any other value skips plot generation.
        - queue (asyncio.Queue): Async queue for real-time progress streaming.

        Returns:
        AsyncGenerator[str, None]: Yields three objects upon completion:
        - performance (pd.DataFrame): Aggregated strategy performance metrics.
        - trade_performance (pd.DataFrame): Detailed trade-by-trade results.
        - stats (pd.DataFrame): Statistical summaries (e.g., Sharpe ratio, max drawdown).

        Side effects:
        - Reads leverage configurations from './app/configs/leverages.yml'.
        - May create HTML plot files in either './app/templates/static/backtest_plots/temp' (temporary)
        or './app/templates/static/backtest_plots' (persistent).
        - Streams logs/results via the provided asyncio.Queue (e.g., for frontend progress updates).

        Notes:
        - The strategy module path must follow the convention: 'app.backbone.strategies.{strategy_name}'.
        - Temporary plots include a timestamp and are automatically purged by a separate cleanup process.
        - Risk-free rate is fetched from the application's config service (used for Sharpe ratio calculations).
        - Margin is calculated as 1/leverage (e.g., 50x leverage → 2% margin requirement).
        """
        await log_message(queue, '', '', "log", f"Loading strategy {strategy.Name}")
        
        strategy_name = strategy.Name.split(".")[1]
        bot_name = f'{strategy_name}_{ticker.Name}_{timeframe.Name}_{risk}'
        
        strategy_path = 'app.backbone.strategies.' + strategy.Name
        strategy_func = load_function(strategy_path)
        
        await log_message(queue, '', '', "log", f"Strategy {strategy.Name} loaded succesfully")
        await log_message(queue, '', '', "log", "Loading leverages")

        with open("./app/configs/leverages.yml", "r") as file_name:
            leverages = yaml.safe_load(file_name)
            await log_message(queue, '', '', "log", "Leverages loaded")

        leverage = leverages[ticker.Name]
        margin = 1 / leverage
        
        await log_message(queue, ticker.Name, timeframe.Name, "log", "Getting data")

        prices = get_data(ticker.Name, timeframe.MetaTraderNumber, date_from, date_to)
        prices.index = pd.to_datetime(prices.index)

        await log_message(queue, ticker.Name, timeframe.Name, "log", f"Data ready: {prices.head(1)}")

        await log_message(queue, ticker.Name, timeframe.Name, "log", "Starting backtest")

        filename = f'{bot_name}_{date_from.strftime("%Y%m%d")}_{date_to.strftime("%Y%m%d")}'
        plot_path = None

        if save_bt_plot == 'temp':
            plot_path = './app/templates/static/backtest_plots/temp'
            await log_message(queue, ticker.Name, timeframe.Name, "link", os.path.join('/backtest_plots/temp', filename + '.html'))

        elif save_bt_plot == 'persist':
            plot_path = './app/templates/static/backtest_plots'

        risk_free_rate = float(self.config_service.get_by_name('RiskFreeRate').Value)

        performance, trade_performance, stats = run_strategy_and_get_performances(
            strategy=strategy_func,
            ticker=ticker,
            timeframe=timeframe,
            risk=risk,
            prices=prices,
            initial_cash=initial_cash,
            risk_free_rate=risk_free_rate,
            margin=margin,
            plot_path=plot_path,
            file_name = filename,
            save_report= save_bt_plot == 'persist' # Solo genera el reporte si no es una corrida temporal
        )

        await log_message(queue, ticker.Name, timeframe.Name, "log", "The backtest is done :)")

        await log_message(queue, ticker.Name, timeframe.Name, "completed", f"{stats.to_string()}")
        
        return performance, trade_performance, stats

    @streaming_endpoint
    async def run_backtests_and_save(
        self,
        initial_cash: float,
        strategies: Strategy | List[Strategy],
        tickers: Ticker | List[Ticker],
        timeframes: List[Timeframe],
        date_from: date,
        date_to: date,
        method: str,
        risks: float | List[float],
        save_bt_plot: str,
        queue: asyncio.Queue,
        ) -> AsyncGenerator[str, None]:
        """
        Executes multiple backtests in parallel for all combinations of strategies, tickers, timeframes and risks,
        then saves results to the database while streaming progress updates.

        This function handles the complete backtesting pipeline:
        - Generates all possible combinations of input parameters
        - Checks for existing backtest results to avoid duplicate work
        - Runs new backtests when needed using run_backtest()
        - Saves performance metrics, trade details, and statistics to the database
        - Provides real-time feedback through an async queue

        Steps performed:
        1. Normalizes all input parameters to lists (single items → single-element lists)
        2. Generates all combinations of strategies/tickers/timeframes/risks
        3. For each combination:
        - Checks if bot exists in database
        - Verifies if backtest already exists for the date range
        - Runs new backtest if needed
        - Converts results to database objects
        - Saves all data (bot, performance, trades) transactionally
        4. Streams progress, warnings, and completion messages via queue

        Parameters:
        - initial_cash (float): Starting capital for all backtests
        - strategies (Strategy|List[Strategy]): Single strategy or list to test
        - tickers (Ticker|List[Ticker]): Financial instrument(s) to backtest
        - timeframes (List[Timeframe]): Time intervals to test (e.g. ['1H', '4H'])
        - date_from (date): Start date for historical data
        - date_to (date): End date for historical data
        - method (str): Backtesting methodology identifier
        - risks (float|List[float]): Risk percentage(s) per trade (e.g. [0.01, 0.02])
        - save_bt_plot (str): Plot saving mode:
            - 'temp': Temporary storage
            - 'persist': Permanent storage
            - Other: Skip plot generation
        - queue (asyncio.Queue): Async queue for progress streaming

        Returns:
        - AsyncGenerator[str, None]: Yields list of saved BotPerformance objects when complete

        Side effects:
        - Creates/updates database records for:
            - Bot configurations
            - Performance metrics
            - Individual trades
        - May generate plot files depending on save_bt_plot parameter
        - Streams messages to provided queue including:
            - Progress logs
            - Warnings about duplicates
            - Completion/failure notifications

        Notes:
        - Automatically skips existing backtests for the same parameters/date range
        - Uses atomic database transactions to ensure data consistency
        - New bots are created if they don't exist in the database
        - Trade history is extracted from backtest stats and saved relationally
        - All database operations use the service's db_service interface
        - Failures for individual combinations don't stop overall execution
        """
        try:
            backtests = []
            combinations = None

            strategies = [strategies] if type(strategies) != list else strategies
            tickers = [tickers] if type(tickers) != list else tickers
            timeframes = [timeframes] if type(timeframes) != list else timeframes
            risks = [risks] if type(risks) != list else risks
  
            combinations = itertools.product(strategies, tickers, timeframes, risks)

            async for combination in self.async_iterator(list(combinations)):
                strategy, ticker, timeframe, risk = combination

                strategy_name = strategy.Name.split(".")[1]

                await log_message(queue, ticker.Name, timeframe.Name, "log", "Starting")
                
                bot_name = f'{strategy_name}_{ticker.Name}_{timeframe.Name}_{risk}'
                
                await log_message(queue, ticker.Name, timeframe.Name, "log", "Checking bot in the database")

                bot = self.bot_service.get_bot(
                    strategy_id=strategy.Id,
                    ticker_id=ticker.Id,
                    timeframe_id=timeframe.Id,
                    risk=risk   
                )
                
                bot_exists = False

                if bot:
                    await log_message(queue, ticker.Name, timeframe.Name, "log", "The bot already exists")

                    bot_exists = True

                    await log_message(queue, ticker.Name, timeframe.Name, "log", "Checking backtest in the database")
                    result_performance = self.get_performances_by_bot_dates(
                        bot_id=bot.Id, 
                        date_from=date_from, 
                        date_to=date_to
                    )
                    
                    if result_performance:
                        backtests.append(result_performance)
                        await log_message(queue, ticker.Name, timeframe.Name, "warning", "The backtest already exists. Skipping...")
                        continue

                else:
                    await log_message(queue, ticker.Name, timeframe.Name, "log", "The bot is not in the database")
                    bot = Bot(
                        Name=bot_name,
                        StrategyId=strategy.Id,
                        TickerId=ticker.Id,
                        TimeframeId=timeframe.Id,
                        Risk=risk
                    )

                try:
                    performance, trade_performance, stats = await self.run_backtest(
                        initial_cash=initial_cash,
                        strategy=strategy,
                        ticker=ticker,
                        timeframe=timeframe,
                        date_from=date_from,
                        date_to=date_to,
                        method=method,
                        risk=risk,
                        save_bt_plot=save_bt_plot,
                        queue=queue,
                    )

                    await log_message(queue, ticker.Name, timeframe.Name, "log", "Saving metrics in the database")

                    bot_performance_for_db = _performance_from_df_to_obj(
                        performance, 
                        date_from, 
                        date_to, 
                        risk, 
                        method, 
                        bot,
                        initial_cash,
                    )
                    
                    trade_performance_for_db = [BotTradePerformance(**row) for _, row in trade_performance.iterrows()].pop()
                    trade_performance_for_db.BotPerformance = bot_performance_for_db 
                    
                    with self.db_service.get_database() as db:
                        if not bot_exists:
                            self.db_service.create(db, bot)
                        
                        backtests.append(self.db_service.create(db, bot_performance_for_db))

                        self.db_service.create(db, trade_performance_for_db)
                        
                        trade_history = trades_from_df_to_obj(stats._trades)

                        for trade in trade_history:
                            trade.BotPerformance = bot_performance_for_db
                            self.db_service.create(db, trade)

                    await log_message(queue, ticker.Name, timeframe.Name, "completed", "It's done buddy ;)")
                
                except Exception as e:
                    await log_message(queue, '', '', "failed", f"{str(e)}")
            
            return backtests
            
        except Exception as e:
            await log_message(queue, '', '', "failed", f"{str(e)}")

    def get_performances_by_strategy_ticker(self, strategy_id, ticker_id) -> BotPerformance:
        with self.db_service.get_database() as db:
            bot_performances = (
                db.query(BotPerformance)
                .join(Bot, Bot.Id == BotPerformance.BotId)
                .filter(Bot.TickerId == ticker_id)
                .filter(Bot.StrategyId == strategy_id)
                .order_by(desc(BotPerformance.StabilityWeightedRar))
                .all()
            )

            return bot_performances

    def get_performances_by_bot_dates(self, bot_id, date_from, date_to) -> BotPerformance:
        with self.db_service.get_database() as db:
            bot_performance = (
                db.query(BotPerformance)
                .options(
                    joinedload(BotPerformance.RandomTest).joinedload(RandomTest.RandomTestPerformance),
                    joinedload(BotPerformance.LuckTest).joinedload(LuckTest.LuckTestPerformance),
                )
                .filter(BotPerformance.BotId == bot_id)
                .filter(BotPerformance.DateFrom == date_from)
                .filter(BotPerformance.DateTo == date_to)
                .first()
            )
            
            return bot_performance
     
    def get_performance_by_bot(self, bot_id) -> BotPerformance: # cambiar bot_id por backtest_id
        with self.db_service.get_database() as db:
            bot_performance = self.db_service.get_many_by_filter(db, BotPerformance, BotId=bot_id)  
            return bot_performance
            
    def get_bot_performance_by_id(self, bot_performance_id) -> BotPerformance: # cambiar bot_id por backtest_id
        with self.db_service.get_database() as db:
            bot_performance = self.db_service.get_by_id(db, BotPerformance, id=bot_performance_id)  

            return bot_performance

    def delete_from_strategy(self, strategy_id) -> OperationResult:
        with self.db_service.get_database() as db:
            bp_subq = (
                select(BotPerformance.Id)
                .join(Bot, BotPerformance.BotId == Bot.Id)
                .where(Bot.StrategyId == strategy_id)
            ).subquery()

            db.execute(
                delete(MetricWharehouse).where(
                    MetricWharehouse.MontecarloTestId.in_(
                        select(MontecarloTest.Id).where(MontecarloTest.BotPerformanceId.in_(bp_subq))
                    )
                )
            )

            db.execute(
                delete(MontecarloTest).where(MontecarloTest.BotPerformanceId.in_(bp_subq))
            )

            db.execute(
                delete(RandomTest).where(RandomTest.BotPerformanceId.in_(bp_subq))
            )

            lucktest_bp_subq = (
                select(LuckTest.LuckTestPerformanceId)
                .where(LuckTest.BotPerformanceId.in_(bp_subq))
            ).subquery()

            bot_performances = db.query(BotPerformance).where(BotPerformance.Id.in_(bp_subq)).all()

            db.execute(delete(BotPerformance).where(BotPerformance.Id.in_(lucktest_bp_subq)))

            db.execute(
                delete(LuckTest).where(LuckTest.BotPerformanceId.in_(bp_subq))
            )

            db.execute(
                delete(BotTradePerformance).where(BotTradePerformance.BotPerformanceId.in_(bp_subq))
            )

            db.execute(
                delete(Trade).where(Trade.BotPerformanceId.in_(bp_subq))
            )

            db.execute(
                delete(BotPerformance).where(BotPerformance.Id.in_(bp_subq))
            )

            db.execute(
                delete(Bot).where(Bot.StrategyId == strategy_id)
            )
        
        # Borrar archivos relacionados
        for bot_performance in bot_performances:
            str_date_from = str(bot_performance.DateFrom).replace('-', '')
            str_date_to = str(bot_performance.DateTo).replace('-', '')
            file_name = f'{bot_performance.Bot.Name}_{str_date_from}_{str_date_to}.html'
            for folder in ['luck_test_plots', 'correlation_plots', 't_test_plots', 'backtest_plots', 'backtest_plots/reports']:
                path = os.path.join('./app/templates/static/', folder, file_name)
                if os.path.exists(path):
                    os.remove(path)

        return OperationResult(ok=True, message=None, item=None)


    def delete(self, bot_performance) -> OperationResult:
        with self.db_service.get_database() as db:
            # Cargar el objeto con relaciones necesarias
            bot_performance_id = bot_performance if isinstance(bot_performance, int) else bot_performance.Id

            bot_performance = db.query(BotPerformance)\
                .options(
                    joinedload(BotPerformance.BotTradePerformance),
                    joinedload(BotPerformance.Bot),
                    joinedload(BotPerformance.LuckTest),
                    joinedload(BotPerformance.RandomTest),
                )\
                .filter(BotPerformance.Id == bot_performance_id)\
                .first()

            if not bot_performance:
                return OperationResult(ok=False, message="BotPerformance not found", item=None)

            # Eliminar relaciones directas
            if bot_performance.BotTradePerformance:
                self.db_service.delete(db, BotTradePerformance, bot_performance.BotTradePerformance.Id)

            # Eliminar Montecarlo y métricas asociadas
            montecarlo_test = self.db_service.get_by_filter(db, MontecarloTest, BotPerformanceId=bot_performance_id)
            if montecarlo_test:
                self.db_service.delete_many_by_filter(db, MetricWharehouse, MontecarloTestId=montecarlo_test.Id)
                self.db_service.delete(db, MontecarloTest, montecarlo_test.Id)

            # Eliminar LuckTests y sus performances
            luck_tests = self.db_service.get_many_by_filter(db, LuckTest, BotPerformanceId=bot_performance_id)
            for lt in luck_tests:
                self.db_service.delete(db, BotPerformance, lt.LuckTestPerformanceId)
                self.db_service.delete(db, LuckTest, lt.Id)

            # Borrar archivos relacionados
            str_date_from = str(bot_performance.DateFrom).replace('-', '')
            str_date_to = str(bot_performance.DateTo).replace('-', '')
            file_name = f'{bot_performance.Bot.Name}_{str_date_from}_{str_date_to}.html'
            for folder in ['luck_test_plots', 'correlation_plots', 't_test_plots', 'backtest_plots', 'backtest_plots/reports']:
                path = os.path.join('./app/templates/static/', folder, file_name)
                if os.path.exists(path):
                    os.remove(path)

            # Eliminar RandomTests y sus dependencias
            if bot_performance.RandomTest:
                rt = bot_performance.RandomTest
                rt_perf = self.db_service.get_by_id(db, BotPerformance, rt.RandomTestPerformanceId)
                if rt_perf:
                    rt_trade_perf = self.db_service.get_by_filter(db, BotTradePerformance, BotPerformanceId=rt_perf.Id)
                    if rt_trade_perf:
                        self.db_service.delete(db, BotTradePerformance, rt_trade_perf.Id)
                    self.db_service.delete(db, BotPerformance, rt.RandomTestPerformanceId)
                self.db_service.delete(db, RandomTest, rt.Id)

            # Borrar trades relacionados
            self.db_service.delete_many_by_filter(db, Trade, BotPerformanceId=bot_performance_id)

            # Si no hay más performance, borrar el Bot
            if bot_performance.BotId:
                rem = db.query(BotPerformance).filter(BotPerformance.BotId == bot_performance.BotId).count()
                if rem == 1:
                    self.db_service.delete(db, Bot, bot_performance.BotId)

            self.db_service.delete(db, BotPerformance, bot_performance.Id)
            self.db_service.save(db)

        return OperationResult(ok=True, message=None, item=None)

    
    def update_favorite(self, performance_id) -> OperationResult:

        with self.db_service.get_database() as db:
            performance = self.db_service.get_by_id(db, BotPerformance, performance_id)
            
            if not performance:
                return OperationResult(ok=False, message='Backtest not found', item=None)
            
            performance.Favorite = not performance.Favorite
            updated_performance = self.db_service.update(db, BotPerformance, performance)   

            if not updated_performance:
                return OperationResult(ok=False, message='Update failed', item=None)

            return OperationResult(ok=True, message=None, item=updated_performance.Favorite)

    def get_robusts(self) -> List[BotPerformance]:
        with self.db_service.get_database() as db:
            # Subquery para filtrar estrategias robustas (RreturnDd promedio > 1)
            subquery = (
                db.query(
                    Bot.StrategyId,
                    Bot.TickerId,
                )
                .join(BotPerformance, Bot.Id == BotPerformance.BotId)
                .join(Timeframe, Bot.TimeframeId == Timeframe.Id)
                .filter(
                    BotPerformance.RreturnDd != "NaN",
                    Timeframe.Selected == True
                )
                .group_by(Bot.StrategyId, Bot.TickerId)
                .having(func.avg(BotPerformance.RreturnDd) > 1)
                .subquery()
            )

            # Alias para evitar ambigüedad
            bot_alias = aliased(Bot)
            bp_alias = aliased(BotPerformance)

            # Subquery para asignar un número de fila basado en StabilityWeightedRar
            ranked_subquery = (
                db.query(
                    bp_alias.Id.label("bp_id"),
                    func.row_number()
                    .over(
                        partition_by=[bot_alias.StrategyId, bot_alias.TickerId],
                        order_by=bp_alias.StabilityWeightedRar.desc(),
                    )
                    .label("row_num"),
                )
                .join(bot_alias, bot_alias.Id == bp_alias.BotId)
                .join(
                    subquery,
                    (bot_alias.StrategyId == subquery.c.StrategyId)
                    & (bot_alias.TickerId == subquery.c.TickerId),
                )
                .subquery()
            )

            # Query final seleccionando solo los mejores (row_num == 1)
            data = (
                db.query(bp_alias)
                .join(ranked_subquery, bp_alias.Id == ranked_subquery.c.bp_id)
                .filter(ranked_subquery.c.row_num == 1)
                .all()
            )

            return data
               
    def get_robusts_by_strategy_id(self, strategy_id) -> List[BotPerformance]:
        with self.db_service.get_database() as db:
            # Subquery para calcular el promedio de RreturnDd por StrategyId y TickerId
            subquery = (
                db.query(
                    Bot.StrategyId,
                    Bot.TickerId,
                )
                .join(BotPerformance, Bot.Id == BotPerformance.BotId)
                .join(Timeframe, Bot.TimeframeId == Timeframe.Id)
                .filter(
                    Bot.StrategyId == strategy_id,
                    BotPerformance.RreturnDd != "NaN",
                    Timeframe.Selected == True,
                )
                .group_by(Bot.StrategyId, Bot.TickerId)
                .having(func.avg(BotPerformance.RreturnDd) >= 1)  # >= en lugar de > para incluir exactamente 1
                .subquery()
            )

            # Alias para evitar ambigüedad en las relaciones
            bot_alias = aliased(Bot)
            bp_alias = aliased(BotPerformance)

            # Subquery para seleccionar la mejor temporalidad por StrategyId - TickerId
            best_temporalities = (
                db.query(
                    bp_alias.BotId,
                    bot_alias.StrategyId,
                    bot_alias.TickerId,
                    func.max(bp_alias.StabilityWeightedRar).label("max_custom_metric")
                )
                .join(bot_alias, bot_alias.Id == bp_alias.BotId)
                .join(subquery, (bot_alias.StrategyId == subquery.c.StrategyId) & (bot_alias.TickerId == subquery.c.TickerId))
                .group_by(bot_alias.StrategyId, bot_alias.TickerId)
                .subquery()
            )

            # Query final para traer solo los registros que corresponden a la mejor temporalidad
            data = (
                db.query(bp_alias)
                .join(bot_alias, bot_alias.Id == bp_alias.BotId)
                .join(best_temporalities,
                    (bot_alias.StrategyId == best_temporalities.c.StrategyId) &
                    (bot_alias.TickerId == best_temporalities.c.TickerId) &
                    (bp_alias.StabilityWeightedRar == best_temporalities.c.max_custom_metric))
                .all()
            )

            return data

    def get_favorites(self) -> List[BotPerformance]:
        with self.db_service.get_database() as db:
            favorites = self.db_service.get_many_by_filter(db, BotPerformance, Favorite=True)
        
        return favorites
        
    def get_trades(self, bot_performance_id:int) -> List[Trade]:
        with self.db_service.get_database() as db:
            bot_performance = self.db_service.get_by_id(db, BotPerformance, id=bot_performance_id)  
            return bot_performance.TradeHistory

    def get_by_filter(
        self,
        return_: float = None,
        drawdown: float = None,
        stability_ratio: float = None,
        sharpe_ratio: float = None,
        trades: float = None,
        rreturn_dd: float = None,
        custom_metric: float = None,
        winrate: float = None,
        strategy: str = None,
        ticker: str = None
    ) -> List[BotPerformance]:
        with self.db_service.get_database() as db:
            filters = []

            # Base query y joins explícitos
            query = db.query(BotPerformance)\
                .join(Bot, Bot.Id == BotPerformance.BotId)\
                .join(Strategy, Strategy.Id == Bot.StrategyId)\
                .join(Ticker, Ticker.Id == Bot.TickerId)

            # Filtros numéricos
            if return_ is not None:
                filters.append(BotPerformance.Return >= return_)
            if drawdown is not None:
                filters.append(BotPerformance.Drawdown <= drawdown)
            if stability_ratio is not None:
                filters.append(BotPerformance.StabilityRatio >= stability_ratio)
            if sharpe_ratio is not None:
                filters.append(BotPerformance.SharpeRatio >= sharpe_ratio)
            if trades is not None:
                filters.append(BotPerformance.Trades >= trades)
            if rreturn_dd is not None:
                filters.append(BotPerformance.RreturnDd >= rreturn_dd)
            if custom_metric is not None:
                filters.append(BotPerformance.StabilityWeightedRar >= custom_metric)
            if winrate is not None:
                filters.append(BotPerformance.WinRate >= winrate)

            # Filtros por texto con ILIKE correctamente aplicados
            if strategy:
                filters.append(Strategy.Name.ilike(f"%{strategy}%"))
            if ticker:
                filters.append(Ticker.Name.ilike(f"%{ticker}%"))

            # Query final que aplica filtros correctamente
            final_query = (
                query
                .filter(*filters)
                .options(
                    joinedload(BotPerformance.Bot),
                    joinedload(BotPerformance.BotTradePerformance),
                    joinedload(BotPerformance.TradeHistory),
                    joinedload(BotPerformance.MontecarloTest),
                    joinedload(BotPerformance.LuckTest),
                    joinedload(BotPerformance.RandomTest),
                )
                .order_by(BotPerformance.RreturnDd.desc())
                .all()
            )

            return final_query
