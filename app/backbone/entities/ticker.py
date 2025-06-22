from sqlalchemy import Column, ForeignKey, Integer, Float, String
from sqlalchemy.orm import relationship

from . import Base

# Clase que representa una tabla en la base de datos
class Ticker(Base):
    __tablename__ = 'Tickers'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)
    CategoryId = Column(Integer, ForeignKey('Categories.Id'))  # Referencia a la tabla Categories
    Name = Column(String, nullable=False) 
    Spread = Column(Float, nullable=False)
    
    Category = relationship('Category', back_populates='Tickers', lazy='joined')
    Bot = relationship('Bot', back_populates='Ticker', lazy='select')

    def __repr__(self):
        return f"<Ticker(id={self.Id}, name='{self.Name}', spread={self.Spread})>"
