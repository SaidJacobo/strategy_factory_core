from sqlalchemy import Column, DateTime, ForeignKey, Integer, Float, Date, Boolean, String
from sqlalchemy.orm import relationship
from . import Base

class Trade(Base):
    __tablename__ = 'Trades'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    BotPerformanceId = Column(Integer, ForeignKey('BotPerformances.Id'), nullable=False)

    Size = Column(Integer, nullable=False)
    EntryBar = Column(Integer, nullable=False)
    ExitBar = Column(Integer, nullable=False)
    EntryPrice = Column(Float, nullable=False)
    ExitPrice = Column(Float, nullable=False)
    SL = Column(Float, nullable=True)
    TP = Column(Float, nullable=True)
    Tag = Column(String, nullable=True)
    PnL = Column(Float, nullable=False)
    NetPnL = Column(Float, nullable=False)
    Commission = Column(Float, nullable=False)
    ReturnPct = Column(Float, nullable=False)
    EntryTime = Column(DateTime, nullable=False)
    ExitTime = Column(DateTime, nullable=False)
    Duration = Column(Integer, nullable=False)
    Equity = Column(Float, nullable=False)
    TopBest = Column(Boolean, nullable=True)
    TopWorst = Column(Boolean, nullable=True)
    EntryConversionRate = Column(Float, nullable=False)
    ExitConversionRate = Column(Float, nullable=False)
    

    # Relación con BotPerformance
    BotPerformance = relationship('BotPerformance', back_populates='TradeHistory', lazy='joined')

    def __repr__(self):
        return f"<TradeHistory(Id={self.Id}, Size={self.Size}, PnL={self.PnL}, ReturnPct={self.ReturnPct})>"
