from typing import List
from app.backbone.database.db_service import DbService
from app.backbone.entities.bot import Bot
from app.backbone.services.operation_result import OperationResult


class BotService:
    def __init__(self):
        self.db_service = DbService()
    
    def get_bots_by_strategy(self, strategy_id) -> List[Bot]:
        with self.db_service.get_database() as db:
            bots = self.db_service.get_many_by_filter(db, Bot, StrategyId=strategy_id)
            return bots
            
    def get_bot(self, strategy_id, ticker_id, timeframe_id, risk) -> Bot:
        with self.db_service.get_database() as db:
            bot = self.db_service.get_by_filter(
                db, 
                Bot, 
                StrategyId=strategy_id,
                TickerId=ticker_id,
                TimeframeId=timeframe_id,
                Risk=risk
            )
            
            return bot

    def get_all_bots(self) -> List[Bot]:
        with self.db_service.get_database() as db:
            all_bots = self.db_service.get_all(db, Bot)
            return all_bots
            
    def get_bot_by_id(self, bot_id) -> Bot:
        with self.db_service.get_database() as db:
            bot = self.db_service.get_by_id(db, Bot, id=bot_id)
            return bot

