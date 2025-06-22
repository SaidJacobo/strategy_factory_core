from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from app.view_models.category_vm import CategoryVM
from app.view_models.strategy_vm import StrategyVM
from app.view_models.timeframe_vm import TimeframeVM

class BacktestCreateVM(BaseModel):
    InitialCash: Optional[int] = None

    Strategies: Optional[List[StrategyVM]] = None
    StrategyId: Optional[str] = None

    Categories: Optional[List[CategoryVM]] = None
    CategoryId: Optional[str] = None
    
    TickerId: Optional[str] = None

    DateFrom: Optional[str] = None
    DateTo: Optional[str] = None
    Risk: Optional[float] = None
    SaveBtPyPlot: Optional[bool] = None

    M1: Optional[bool] = False
    M2: Optional[bool] = False
    M3: Optional[bool] = False
    M4: Optional[bool] = False
    M5: Optional[bool] = False
    M6: Optional[bool] = False
    M10: Optional[bool] = False
    M15: Optional[bool] = False
    M12: Optional[bool] = False
    M20: Optional[bool] = False
    M30: Optional[bool] = False

    H1: Optional[bool] = False
    H2: Optional[bool] = False
    H3: Optional[bool] = False
    H4: Optional[bool] = False
    H6: Optional[bool] = False
    H8: Optional[bool] = False
    H12: Optional[bool] = False

    D1: Optional[bool] = False
    W1: Optional[bool] = False
    MN1: Optional[bool] = False