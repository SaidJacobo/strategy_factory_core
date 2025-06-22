from uuid import UUID
from pydantic import BaseModel, ConfigDict


class StrategyVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id:int
    Name: str
    Description: str
    MetaTraderName: str