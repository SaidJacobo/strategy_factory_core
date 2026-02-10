from pydantic import BaseModel, ConfigDict


class PortfolioBacktestVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    BacktestId:int
    BotName: str
    BotWeight: float