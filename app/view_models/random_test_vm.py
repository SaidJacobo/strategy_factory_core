from typing import Optional
from pydantic import BaseModel, ConfigDict
from app.view_models.performance_metrics_vm import PerformanceMetricsVM

class RandomTestVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    Id: int
    BotPerformanceId: int
    RandomTestPerformanceId: Optional[int] = None
    Iterations: int
    ReturnDdMeanDiff: float
    ReturnDdStdDiff: float
    ReturnDdPValue: float
    ReturnDdZScore: float

    ReturnMeanDiff: float
    ReturnStdDiff: float
    ReturnPValue: float
    ReturnZScore: float

    DrawdownMeanDiff: float
    DrawdownStdDiff: float
    DrawdownPValue: float
    DrawdownZScore: float

    WinrateMeanDiff: float
    WinrateStdDiff: float
    WinratePValue: float
    WinrateZScore: float
    # RandomTestPerformance: Optional[PerformanceMetricsVM] = None
