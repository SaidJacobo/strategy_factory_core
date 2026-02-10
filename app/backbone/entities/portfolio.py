from sqlalchemy import Column, Integer, String
from . import Base
from sqlalchemy.orm import relationship

# Clase que representa una tabla en la base de datos
class Portfolio(Base):
    __tablename__ = 'Portfolios'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False)
    Description = Column(String, nullable=True)

    # Relación con PortfolioBacktest
    PortfolioBacktests = relationship("PortfolioBacktest", back_populates="Portfolio", lazy="select")
    SystemPortfolios = relationship("SystemPortfolio", back_populates="Portfolio", lazy="joined")

    GroupedMetrics = relationship("GroupedMetrics", back_populates="Portfolio", lazy="joined", uselist=False)

    def __repr__(self):
        return f"<Portfolio(id={self.Id}, name='{self.Name}', description={self.Description})>"
