import os
from typing import List
from app.backbone.entities.bot_performance import BotPerformance
import numpy as np
import pandas as pd
from app.backbone.entities.grouped_metrics import GroupedMetrics
from app.backbone.entities.portfolio import Portfolio
from app.backbone.entities.portfolio_backtest import PortfolioBacktest
from app.backbone.entities.system_portfolio import SystemPortfolio
from app.backbone.services.backtest_service import BacktestService
from app.backbone.database.db_service import DbService
from app.backbone.services.config_service import ConfigService
from app.backbone.services.operation_result import OperationResult
from app.backbone.services.utils import FtmoChallengeMetrics, MarginMetrics, calculate_margin_metrics, calculate_sharpe_ratio, calculate_stability_ratio, ftmo_simulator, get_trade_df_from_db, get_portfolio_equity_curve, max_drawdown
import plotly.graph_objects as go
from collections import namedtuple
from scipy.stats import jarque_bera, skew, kurtosis

    
Metrics = namedtuple('Metrics',[
    'sharpe_ratio',
    'mean_returns',
    'std_returns',
    'stability_ratio',
    'return_',
    'dd', 
    'return_dd',
    'jarque_bera_stat', 
    'jarque_bera_p_value',
    'skew',
    'kurtosis'
])

class PortfolioService:
    def __init__(self):
        self.db_service = DbService()
        self.backtest_service = BacktestService()
        self.config_service = ConfigService()
        
    def create( self, name:str, description:str) -> OperationResult:
        with self.db_service.get_database() as db:
            
            portfolio_by_filter = self.db_service.get_by_filter(db, Portfolio, Name=name)
            
            if portfolio_by_filter is None:
                
                new_portfolio = Portfolio(Name=name, Description=description)
                
                portfolio = self.db_service.create(db, new_portfolio)

                result = OperationResult(ok=True, message=None, item=portfolio)
                
                return result
            
            result = OperationResult(ok=False, message='El item ya esta cargado en la BD', item=None)
            return result

    def update(self, id:int, name:str, description:str) -> OperationResult:
        with self.db_service.get_database() as db:
            new_portfolio = Portfolio(Id=id, Name=name, Description=description)
            portfolio = self.db_service.update(db, Portfolio, new_portfolio)
            
            result = OperationResult(ok=True, message=None, item=portfolio)
            return result

    def delete(self, portfolio_id:int):
        with self.db_service.get_database() as db:
            portfolio = self.db_service.get_by_id(db, Portfolio, id=portfolio_id)

            for sys_pf in portfolio.PortfolioBacktests:
                self.db_service.delete(db, PortfolioBacktest, sys_pf.Id)

            for sys_pf in portfolio.SystemPortfolios:
                self.db_service.delete(db, SystemPortfolio, sys_pf.Id)

            if portfolio.GroupedMetrics:
                self.db_service.delete(db, GroupedMetrics, portfolio.GroupedMetrics.Id)

            self.db_service.delete(db, Portfolio, portfolio.Id)

        plot_path = f'./app/templates/static/portfolio_plots/{portfolio_id}.html'
        if os.path.exists(plot_path):
            os.remove(plot_path)

    def get_all(self) -> List[Portfolio]:
        with self.db_service.get_database() as db:
            all_portfolios = self.db_service.get_all(db, Portfolio)
            return all_portfolios
    
    def get_by_id(self, portfolio_id:int) -> Portfolio:
        with self.db_service.get_database() as db:
            portfolio = self.db_service.get_by_id(db, Portfolio, id=portfolio_id)
            return portfolio
    
    def get_portfolio_backtest(self, portfolio_id:int, bot_performance_id:int) -> PortfolioBacktest:
        with self.db_service.get_database() as db:
            portfolio_backtest = self.db_service.get_by_filter(
                db, PortfolioBacktest, PortfolioId=portfolio_id, BotPerformanceId=bot_performance_id
            )
            
            return portfolio_backtest
   
    def add_performance(self, portfolio_id: int, bot_performance_id: int, weight:float=None) -> OperationResult:
        # Chequeo previo: ¿ya existe esa relación?
        result = self.get_portfolio_backtest(
            portfolio_id=portfolio_id,
            bot_performance_id=bot_performance_id
        )
        
        if result:
            return OperationResult(
                ok=False,
                message='This bot is already in the portfolio',
                item=None
            )

        # Abrimos una única sesión para todo el flujo
        with self.db_service.get_database() as db:
            # Cargamos las entidades dentro de esta sesión
            portfolio = db.query(Portfolio).get(portfolio_id)
            bot_performance = db.query(BotPerformance).get(bot_performance_id)

            if not portfolio:
                return OperationResult(
                    ok=False,
                    message='Portfolio not found',
                    item=None
                )
            
            if not bot_performance:
                return OperationResult(
                    ok=False,
                    message='Bot Performance not found',
                    item=None
                )

            # Creamos la relación intermedia
            portfolio_backtest = PortfolioBacktest(
                PortfolioId=portfolio.Id,
                BotPerformanceId=bot_performance.Id,
                Portfolio=portfolio,
                BotPerformance=bot_performance,
                Weight=weight
            )

            _ = self.db_service.create(db, portfolio_backtest)

            return OperationResult(ok=True, message=None, item=None)
        
    def delete_performance(self, portfolio_id:int, bot_performance_id:int) -> OperationResult:
        portfolio_backtest_result = self.get_portfolio_backtest(portfolio_id=portfolio_id, bot_performance_id=bot_performance_id)
        
        if not portfolio_backtest_result:
            return OperationResult(ok=False, message='This bot is not in the portfolio', item=None)
        
        with self.db_service.get_database() as db:
            self.db_service.delete(db, PortfolioBacktest, portfolio_backtest_result.Id)
            return OperationResult(ok=True, message=None, item=None)

    def get_backtests_from_portfolio(self, portfolio_id:int) -> List[BotPerformance]:
        
        with self.db_service.get_database() as db:
            portfolio_backtests = self.db_service.get_many_by_filter(db, PortfolioBacktest, PortfolioId=portfolio_id)
            
            backtests = [portfolio_backtest.BotPerformance for portfolio_backtest in portfolio_backtests ]
            
            return backtests
        
    def get_portfolio_backtests(self, portfolio_id:int) -> List[PortfolioBacktest]:
        with self.db_service.get_database() as db:
            portfolio_backtests = self.db_service.get_many_by_filter(db, PortfolioBacktest, PortfolioId=portfolio_id)
            
            return portfolio_backtests
        
    def get_df_trades(self, portfolio_id: int) -> dict[str, pd.DataFrame]:
        """Obtiene las curvas de equity de cada bot en un portafolio."""
        backtests = self.get_backtests_from_portfolio(portfolio_id)

        trades_with_equity = {
            backtest.Bot.Name: get_trade_df_from_db(backtest.TradeHistory, backtest.Id) 
            for backtest in backtests
        }
        
        return trades_with_equity
    
    def set_bot_weights(self, portfolio_id: int, bot_performance_ids: int | List[int], weights: float | List[float]):
        
        bot_performance_ids = [bot_performance_ids] if not type(bot_performance_ids) == list else bot_performance_ids
        weights = [weights] if not type(weights) == list else weights

        message = None
        with self.db_service.get_database() as db:
            pf_backtests = self.db_service.get_many_by_filter(db, PortfolioBacktest, PortfolioId=portfolio_id)

            for bot_performance_id, weight in zip(bot_performance_ids, weights):
                
                bot_performance = [pf_bt for pf_bt in pf_backtests if pf_bt.BotPerformanceId == bot_performance_id].pop()

                if not bot_performance:
                    message = '' if not message else message
                    message += f'bot_performance_id {bot_performance_id} is not in portfolio {portfolio_id}'
                
                bot_performance.Weight = weight
                self.db_service.update(db, PortfolioBacktest, bot_performance)

        return OperationResult(ok=True, message=message, item=None)