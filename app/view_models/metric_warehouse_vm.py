from pydantic import BaseModel, ConfigDict


class MetricWarehouseVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id: int
    MontecarloTestId: int
    Method: str
    Metric: str
    ColumnName: str
    Value: float