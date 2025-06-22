from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict


class ConfigUpdateVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    InitialCash: float
    DateFromBacktest: str
    DateToBacktest: str
    TelegramBotToken: str
    TelegramBotChatId: str
    PositiveHitThreshold: float
    NegativeHitThreshold: float
    MontecarloIterations: int
    MontecarloRiskOfRuin: float
    RandomTestIterations: int
    LuckTestPercentTradesToRemove:float
    RiskFreeRate: float

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
