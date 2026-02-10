from typing import List
from pydantic import BaseModel, ConfigDict
from app.view_models.system_portfolio_vm import SystemPortfolioVM


class UpdateSystemPortfolioVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    SystemId:int
    SystemWeights: List[SystemPortfolioVM]