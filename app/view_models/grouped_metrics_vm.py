from pydantic import BaseModel, ConfigDict

class GroupedMetricsVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    SharpeRatio: float
    StabilityRatio: float
    Return: float
    Drawdown: float
    RreturnDd: float
    PositiveHits: int
    NegativeHits: int
    SuccessRatio:float
    MeanTimeToPositive: float
    MeanTimeToNegative: float
    StdTimeToPositive: float
    StdTimeToNegative: float
    MarginCalls: int
    StopOuts: int

    JarqueBeraStat: float
    JarqueBeraPValue: float
    Skew: float
    Kurtosis: float
    MeanReturns: float
    StdReturns: float