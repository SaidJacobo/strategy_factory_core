from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from app.view_models.ticker_vm import TickerVM

class CategoryVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    Id:int
    Name: str
    Commission: float
    Tickers: Optional[List[TickerVM]] = None  # Relación con Tickers