from sqlalchemy import Column, ForeignKey, Integer, Float, String, Date, Boolean
from sqlalchemy.orm import relationship
from . import Base

class GroupedMetrics(Base):
    __tablename__ = 'GroupedMetrics'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    SystemId = Column(Integer, ForeignKey('Systems.Id'), nullable=True)
    PortfolioId = Column(Integer, ForeignKey('Portfolios.Id'), nullable=True)

    Return = Column(Float, nullable=False)
    Drawdown = Column(Float, nullable=False)
    RreturnDd = Column(Float, nullable=False)
    SharpeRatio = Column(Float, nullable=True)
    JarqueBeraStat = Column(Float, nullable=True)
    JarqueBeraPValue = Column(Float, nullable=True)
    Skew = Column(Float, nullable=True)
    Kurtosis = Column(Float, nullable=True)
    PositiveHits = Column(Float, nullable=True)
    NegativeHits = Column(Float, nullable=True)
    SuccessRatio = Column(Float, nullable=True)
    MeanTimeToPositive = Column(Float, nullable=True)
    MeanTimeToNegative = Column(Float, nullable=True)
    StdTimeToPositive = Column(Float, nullable=True)
    StdTimeToNegative = Column(Float, nullable=True)
    MeanReturns = Column(Float, nullable=True)
    StdReturns = Column(Float, nullable=True)
    MarginCalls = Column(Float, nullable=True)
    StopOuts = Column(Float, nullable=True)
    StabilityRatio = Column(Float, nullable=True)

    System = relationship('System', back_populates='GroupedMetrics', lazy='joined')
    Portfolio = relationship('Portfolio', back_populates='GroupedMetrics', lazy='joined')


    def __repr__(self):
        return f"<GroupedMetrics(Id={self.Id}, SystemId={self.SystemId}, PortfolioId={self.PortfolioId})>"