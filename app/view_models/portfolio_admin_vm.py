from typing import List
from pydantic import BaseModel, ConfigDict

from app.view_models.performance_metrics_vm import PerformanceMetricsVM

class PortfolioAdminVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    FavoritesAndRobusts: List[PerformanceMetricsVM]
    Selected: List[PerformanceMetricsVM] = None