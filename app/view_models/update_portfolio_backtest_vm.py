from typing import List
from pydantic import BaseModel, ConfigDict

from app.view_models.portfolio_backtest_vm import PortfolioBacktestVM


class UpdatePortfolioBacktestVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    PortfolioId:int
    PortfolioBacktests: List[PortfolioBacktestVM]