from pydantic import BaseModel, ConfigDict

from app.view_models.ticker_vm import TickerVM

class BotVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    Id:int
    TickerId: int
    StrategyId: int
    Name: str
    Risk: float
    Ticker: TickerVM