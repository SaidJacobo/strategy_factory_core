from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from app.view_models.grouped_metrics_vm import GroupedMetricsVM
from app.view_models.portfolio_vm import PortfolioVM

class SystemVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id:int
    Name: str
    Description: str
    GroupedMetrics: Optional[GroupedMetricsVM]=None
    Portfolios: List[PortfolioVM]=None
    HasEquityPlot:bool=False