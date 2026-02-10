from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from app.view_models.grouped_metrics_vm import GroupedMetricsVM
from app.view_models.performance_metrics_vm import PerformanceMetricsVM


class PortfolioVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id:int
    Name: str
    Description: Optional[str]=None
    GroupedMetrics: Optional[GroupedMetricsVM]=None
    BotPerformances: List[PerformanceMetricsVM]=None
    HasEquityPlot:bool=False
