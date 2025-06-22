from sqlalchemy import Column, ForeignKey, Float, Integer, String
from sqlalchemy.orm import relationship

from . import Base

# Clase que representa una tabla en la base de datos
class MetricWharehouse(Base):
    __tablename__ = 'MetricsWarehouse'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    MontecarloTestId = Column(Integer, ForeignKey('MontecarloTests.Id'), nullable=False)  # Relación con MontecarloTest

    Method = Column(String, nullable=False)
    Metric = Column(String, nullable=False)
    ColumnName = Column(String, nullable=False)
    Value = Column(Float, nullable=False)

    # Relación con MontecarloTest
    MontecarloTest = relationship('MontecarloTest', back_populates='Metrics', lazy='joined')

    def __repr__(self):
        return f"<MetricWharehouse(id={self.Id}, Method='{self.Method}', Metric={self.Metric}, Value={self.Value})>"
