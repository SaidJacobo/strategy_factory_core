from typing import Optional
from pydantic import BaseModel, ConfigDict
from datetime import date

from app.view_models.bot_trade_performance_vm import BotTradePerformamceVM
from app.view_models.bot_vm import BotVM


class PerformanceMetricsVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Bot: Optional[BotVM] = None
    Id: int = None
    BotId: Optional[int] = None
    DateFrom: date = None
    DateTo: date = None
    Method: str = None
    SharpeRatio: float = None
    StabilityRatio: float = None
    Trades: float = None
    Return: float = None
    Drawdown: float = None
    RreturnDd: float = None
    StabilityWeightedRar: float = None
    WinRate: float = None
    Duration: int = None
    Favorite: bool = False
    InitialCash: float  = None
    ExposureTime: float = None
    KellyCriterion: float = None
    WinratePValue: float = None
    SharpeRatio: Optional[float]
    JarqueBeraStat: Optional[float]
    JarqueBeraPValue: Optional[float]
    Skew: Optional[float]
    Kurtosis: Optional[float]

    BotTradePerformance: Optional[BotTradePerformamceVM] = None
    