from typing import Optional
from pydantic import BaseModel, ConfigDict
from datetime import date, datetime

class TradeVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    Id: int
    BotPerformanceId: int
    Size: float
    EntryBar: int
    ExitBar: int
    EntryPrice: float
    ExitPrice: float
    PnL: float
    NetPnL: float
    ReturnPct: float
    EntryTime: datetime
    ExitTime: datetime
    Duration: int
    Equity: float
    Commission: float
    TopBest: Optional[bool] = None
    TopWorst: Optional[bool] = None