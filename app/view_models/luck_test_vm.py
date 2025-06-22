from pydantic import BaseModel, ConfigDict
from app.view_models.performance_metrics_vm import PerformanceMetricsVM

class LuckTestVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    Id: int
    BotPerformanceId: int
    LuckTestPerformanceId: int
    TradesPercentToRemove: float
    
    LuckTestPerformance: PerformanceMetricsVM

