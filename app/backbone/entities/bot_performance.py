from sqlalchemy import Column, ForeignKey, Integer, Float, String, Date, Boolean
from sqlalchemy.orm import relationship
from . import Base

class BotPerformance(Base):
    __tablename__ = 'BotPerformances'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    BotId = Column(Integer, ForeignKey('Bots.Id'), nullable=True)
    DateFrom = Column(Date, nullable=False, index=True)
    DateTo = Column(Date, nullable=False, index=True)
    Method = Column(String, nullable=False)
    StabilityRatio = Column(Float, nullable=False)
    Trades = Column(Integer, nullable=False)
    Return = Column(Float, nullable=False)
    Drawdown = Column(Float, nullable=False)
    RreturnDd = Column(Float, nullable=False)
    StabilityWeightedRar = Column(Float, nullable=False)
    WinRate = Column(Float, nullable=False)
    Duration = Column(Integer, nullable=False)
    Favorite = Column(Boolean, nullable=False, default=False)
    InitialCash = Column(Float, nullable=False)

    ExposureTime = Column(Float, nullable=False)
    KellyCriterion = Column(Float, nullable=False)
    WinratePValue = Column(Float, nullable=False)
    SharpeRatio = Column(Float, nullable=True)

    JarqueBeraStat = Column(Float, nullable=True)
    JarqueBeraPValue = Column(Float, nullable=True)
    Skew = Column(Float, nullable=True)
    Kurtosis = Column(Float, nullable=True)

    # Relación con otras tablas
    Bot = relationship('Bot', back_populates='BotPerformance', lazy='joined')
    
    BotTradePerformance = relationship('BotTradePerformance', back_populates='BotPerformance', lazy='joined', uselist=False)
    TradeHistory = relationship('Trade', back_populates='BotPerformance', lazy='joined')
    MontecarloTest = relationship('MontecarloTest', back_populates='BotPerformance', lazy='joined', uselist=False)
    LuckTest = relationship('LuckTest', foreign_keys='LuckTest.BotPerformanceId', back_populates='BotPerformance', lazy='joined', uselist=False)
    RandomTest = relationship('RandomTest', foreign_keys='RandomTest.BotPerformanceId', back_populates='BotPerformance', lazy='joined', uselist=False)

    def __repr__(self):
        return f"<BotPerformance(Id={self.Id}, Trades={self.Trades}, Return={self.Return}, Drawdown={self.Drawdown})>"