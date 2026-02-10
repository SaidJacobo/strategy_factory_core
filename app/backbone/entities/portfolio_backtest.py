from sqlalchemy import Column, Float, ForeignKey, Integer
from . import Base
from sqlalchemy.orm import relationship

# Clase que representa una tabla en la base de datos
class PortfolioBacktest(Base):
    __tablename__ = 'PortfoliosBacktests'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)

    PortfolioId = Column(Integer, ForeignKey('Portfolios.Id'), index=True)
    BotPerformanceId = Column(Integer, ForeignKey('BotPerformances.Id'))
    Portfolio = relationship("Portfolio", back_populates="PortfolioBacktests", lazy="joined")
    BotPerformance = relationship("BotPerformance", back_populates="PortfolioBacktests", lazy="joined")

    Weight = Column(Float, nullable=True)


    def __repr__(self):
        return f"<Strategy(id={self.Id}, PortfolioId='{self.PortfolioId}', BotPerformanceId={self.BotPerformanceId})>"
