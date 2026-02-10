
from sqlalchemy import Column, Float, ForeignKey, Integer
from . import Base
from sqlalchemy.orm import relationship

# Clase que representa una tabla en la base de datos
class SystemPortfolio(Base):
    __tablename__ = 'SystemPortfolios'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)

    SystemId = Column(Integer, ForeignKey('Systems.Id'))
    PortfolioId = Column(Integer, ForeignKey('Portfolios.Id'), index=True)
    
    System = relationship("System", back_populates="SystemPortfolios", lazy="joined")
    Portfolio = relationship("Portfolio", back_populates="SystemPortfolios", lazy="joined")

    Weight = Column(Float, nullable=True)


    def __repr__(self):
        return f"<Strategy(id={self.Id}, SystemId='{self.SystemId}', PortfolioId={self.PortfolioId})>"
