from pydantic import BaseModel, ConfigDict

class BotTradePerformamceVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    Id: int
    BotPerformanceId: int
    MeanWinningReturnPct: float
    StdWinningReturnPct: float
    MeanLosingReturnPct: float
    StdLosingReturnPct: float
    MeanTradeDuration: float
    StdTradeDuration: float
    
    LongWinrate: float
    WinLongMeanReturnPct: float
    WinLongStdReturnPct: float
    LoseLongMeanReturnPct: float
    LoseLongStdReturnPct: float
    
    ShortWinrate: float
    WinShortMeanReturnPct: float
    WinShortStdReturnPct: float
    LoseShortMeanReturnPct: float
    LoseShortStdReturnPct: float

    MeanReturnPct: float
    StdReturnPct: float
    ProfitFactor: float
    WinRate: float
    ConsecutiveWins: float
    ConsecutiveLosses: float
    LongCount: float
    ShortCount: float

    LongMeanReturnPct: float
    LongStdReturnPct: float
    ShortMeanReturnPct: float
    ShortStdReturnPct: float