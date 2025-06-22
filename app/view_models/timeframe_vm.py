from typing import Optional
from pydantic import BaseModel, ConfigDict

class TimeframeVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id:int
    Name: str
    MetaTraderNumber: int
    Selected: Optional[bool] = None