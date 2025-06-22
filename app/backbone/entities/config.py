from sqlalchemy import Column, Integer, String
from . import Base
from sqlalchemy.orm import relationship

# Clase que representa una tabla en la base de datos
class Config(Base):
    __tablename__ = 'Configs'  # Nombre de la tabla en la BD

    Id = Column(Integer, primary_key=True, autoincrement=True)
    Name = Column(String, nullable=False) 
    Value = Column(String, nullable=False)


    def __repr__(self):
        return f"<Config(id={self.Id}, name='{self.Name}', Value={self.Value})>"
