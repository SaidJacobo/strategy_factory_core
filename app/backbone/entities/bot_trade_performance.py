from sqlalchemy import Integer, Column, ForeignKey, Float
from sqlalchemy.orm import relationship
from . import Base           

class BotTradePerformance(Base):
    __tablename__ = 'BotTradePerformances'
    
    Id = Column(Integer, primary_key=True, autoincrement=True)
    BotPerformanceId =  Column(Integer, ForeignKey('BotPerformances.Id'), unique=True)
    MeanWinningReturnPct = Column(Float, nullable=False)
    StdWinningReturnPct = Column(Float, nullable=False)
    MeanLosingReturnPct = Column(Float, nullable=False)
    StdLosingReturnPct = Column(Float, nullable=False)
    MeanTradeDuration = Column(Float, nullable=False)
    StdTradeDuration = Column(Float, nullable=False)
    LongWinrate = Column(Float, nullable=False)
    WinLongMeanReturnPct = Column(Float, nullable=False)
    WinLongStdReturnPct = Column(Float, nullable=False)
    LoseLongMeanReturnPct = Column(Float, nullable=False)
    LoseLongStdReturnPct = Column(Float, nullable=False)
    ShortWinrate = Column(Float, nullable=False)
    WinShortMeanReturnPct = Column(Float, nullable=False)
    WinShortStdReturnPct = Column(Float, nullable=False)
    LoseShortMeanReturnPct = Column(Float, nullable=False)
    LoseShortStdReturnPct = Column(Float, nullable=False)

    MeanReturnPct = Column(Float, nullable=False)
    StdReturnPct = Column(Float, nullable=False)
    ProfitFactor = Column(Float, nullable=False)
    WinRate = Column(Float, nullable=False)
    ConsecutiveWins = Column(Integer, nullable=False)
    ConsecutiveLosses = Column(Integer, nullable=False)
    LongCount = Column(Integer, nullable=False)
    ShortCount = Column(Integer, nullable=False)
    LongMeanReturnPct = Column(Integer, nullable=False)
    LongStdReturnPct = Column(Integer, nullable=False)
    ShortMeanReturnPct = Column(Integer, nullable=False)
    ShortStdReturnPct = Column(Integer, nullable=False)
    
    BotPerformance = relationship('BotPerformance', back_populates='BotTradePerformance', lazy='joined')
    
    def __repr__(self):
        return f"<BotTradePerformance(Id={self.Id}, MeanWinningReturnPct={self.MeanWinningReturnPct}, MeanTradeDuration={self.MeanTradeDuration})>"
    
    