from sqlalchemy import Column, ForeignKey, Integer, Float, String
from . import Base
from sqlalchemy.orm import relationship

# Clase que representa una tabla en la base de datos
class Bot(Base):
    __tablename__ = 'Bots'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)
    StrategyId = Column(Integer, ForeignKey('Strategies.Id'))
    TickerId = Column(Integer, ForeignKey('Tickers.Id'))
    TimeframeId = Column(Integer, ForeignKey('Timeframes.Id'))
    Name = Column(String, nullable=False)
    Risk = Column(Float, nullable=False)
    
    Strategy = relationship('Strategy', back_populates='Bot', lazy='joined')
    Ticker = relationship('Ticker', back_populates='Bot', lazy='joined')
    Timeframe = relationship('Timeframe', back_populates='Bot', lazy='joined')
    BotPerformance = relationship('BotPerformance', back_populates='Bot', lazy='joined')
    

    def __repr__(self):
        return f"<Bot(id={self.Id}, strategy_name='{self.Strategy.Id}'>"
    
