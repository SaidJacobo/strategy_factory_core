from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship

from . import Base

# Clase que representa una tabla en la base de datos
class Timeframe(Base):
    __tablename__ = 'Timeframes'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False)
    MetaTraderNumber = Column(Integer, nullable=False)
    Selected = Column(Boolean, nullable=True)

    Bot = relationship('Bot', back_populates='Timeframe', lazy='select')
    
    
    def __repr__(self):
        return f"<Timeframe(id={self.Id}, Name='{self.Name}', MetaTraderNumber={self.MetaTraderNumber})>"
    
