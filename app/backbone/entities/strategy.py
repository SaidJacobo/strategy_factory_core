from sqlalchemy import Column, Integer, String
from . import Base
from sqlalchemy.orm import relationship

# Clase que representa una tabla en la base de datos
class Strategy(Base):
    __tablename__ = 'Strategies'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False) 
    Description = Column(String, nullable=False)
    MetaTraderName = Column(String(length=16), nullable=False)

    Bot = relationship('Bot', back_populates='Strategy', lazy='select')


    def __repr__(self):
        return f"<Strategy(id={self.Id}, name='{self.Name}', description={self.Description})>"
