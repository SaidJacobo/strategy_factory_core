from pydantic import BaseModel, ConfigDict

class SystemPortfolioVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    PortfolioId:int
    PortfolioName: str
    PortfolioWeight: float