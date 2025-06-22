from pydantic import BaseModel, ConfigDict

class TickerVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id:int
    CategoryId:int
    Name: str
    Spread: float