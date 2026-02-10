
from sqlalchemy import Column, Integer, String
from . import Base
from sqlalchemy.orm import relationship

# Clase que representa una tabla en la base de datos
class System(Base):
    __tablename__ = 'Systems'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False)
    Description = Column(String, nullable=False)

    # Relación con PortfolioBacktest
    SystemPortfolios = relationship("SystemPortfolio", back_populates="System", lazy="select")
    GroupedMetrics = relationship("GroupedMetrics", back_populates="System", lazy="joined", uselist=False)

    def __repr__(self):
        return f"<System(id={self.Id}, name='{self.Name}', description={self.Description})>"
