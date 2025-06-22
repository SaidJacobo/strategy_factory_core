from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship

from . import Base

# Clase que representa una tabla en la base de datos
class Category(Base):
    __tablename__ = 'Categories'  # Cambié el nombre para evitar colisión con Tickers
    
    Id = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False) 
    Commission = Column(Float, nullable=False) 
    
    Tickers = relationship('Ticker', back_populates='Category', lazy='joined')

    def __repr__(self):
        return f"<Category(id={self.Id}, name='{self.Name}')>"
