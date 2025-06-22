from typing import List, Optional
from pydantic import BaseModel
from app.view_models.category_vm import CategoryVM
from app.view_models.strategy_vm import StrategyVM
from app.view_models.timeframe_vm import TimeframeVM

class BacktestStrategyCreateVM(BaseModel):

    InitialCash: Optional[float] = None

    DateFrom: Optional[str] = None
    DateTo: Optional[str] = None

    Categories: Optional[List[CategoryVM]] = None
    Strategies: Optional[List[StrategyVM]] = None
    Timeframes: Optional[List[TimeframeVM]] = None
    
    StrategyId: Optional[int] = None
    CategoryId: Optional[int] = None
    TickerId: Optional[int] = None
    TimeframeId: Optional[int] = None

    Code: Optional[str] = None

    Risk: Optional[float] = None
