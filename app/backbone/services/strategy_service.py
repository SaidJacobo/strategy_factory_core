from typing import List
from sqlalchemy import UUID
from app.backbone.entities.bot import Bot
from app.backbone.database.db_service import DbService
from app.backbone.entities.strategy import Strategy
from app.backbone.services.backtest_service import BacktestService
from app.backbone.services.bot_service import BotService
from app.backbone.services.operation_result import OperationResult
from app.backbone.utils.general_purpose import profile_function

class StrategyService:
    def __init__(self):
        self.db_service = DbService()
        self.backtest_service = BacktestService()
        self.bot_service = BotService()

        
    def create(self, name:str, description:str, metatrader_name:str) -> OperationResult:
        with self.db_service.get_database() as db:
            
            strategy_by_filter = self.db_service.get_by_filter(db, Strategy, Name=name)
            
            if strategy_by_filter is None:
                new_strategy = Strategy(Name=name, Description=description, MetaTraderName=metatrader_name)
                strategy = self.db_service.create(db, new_strategy)
                result = OperationResult(ok=True, message=None, item=strategy)
                
                return result
            
            result = OperationResult(ok=False, message='El item ya esta cargado en la BD', item=None)
            return result
            
    def get_all(self) -> List[Strategy]:
        with self.db_service.get_database() as db:
            all_strategies = self.db_service.get_all(db, Strategy)
            return all_strategies

    def delete(self, strategy_id) -> OperationResult:
        bots = self.bot_service.get_bots_by_strategy(strategy_id=strategy_id)
        for bot in bots:
            for backtest in bot.BotPerformance:
                _ = self.backtest_service.delete(backtest.Id)

        with self.db_service.get_database() as db:
        
            strategy = self.db_service.delete(db, Strategy, strategy_id)
            result = OperationResult(ok=True, message=None, item=strategy)
            return result
    
    def get_by_id(self, id) -> Strategy:
        with self.db_service.get_database() as db:
            strategy = self.db_service.get_by_id(db, Strategy, id)
            return strategy
    
    def update(self, id:int, name:str, description:str, metatrader_name:str) -> OperationResult:
        with self.db_service.get_database() as db:
            new_strategy = Strategy(Id=id, Name=name, Description=description, MetaTraderName=metatrader_name)
            
            strategy = self.db_service.update(db, Strategy, new_strategy)
            
            result = OperationResult(ok=True, message=None, item=strategy)
            return result

    def get_used_strategies(self) -> List[Strategy]:
        with self.db_service.get_database() as db:
            strategies = (
                db.query(Strategy)
                .join(Bot, Strategy.Id == Bot.StrategyId)  # Relación entre Strategy y Bot
                .distinct()  # Evita duplicados
                .all()  # Recupera los resultados
            )
            
            return strategies
