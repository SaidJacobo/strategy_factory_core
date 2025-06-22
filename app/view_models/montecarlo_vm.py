from typing import List
from pydantic import BaseModel, ConfigDict
from app.view_models.metric_warehouse_vm import MetricWarehouseVM


class MontecarloVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id: int
    BotPerformanceId: int
    Simulations: int
    ThresholdRuin: float
    Metrics: List[MetricWarehouseVM]
    