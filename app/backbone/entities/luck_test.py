from sqlalchemy import Column, ForeignKey, Integer, Float, String, Date
from sqlalchemy.orm import relationship

from . import Base

class LuckTest(Base):
    __tablename__ = 'LuckTests'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    BotPerformanceId = Column(Integer, ForeignKey('BotPerformances.Id'), nullable=False)
    LuckTestPerformanceId = Column(Integer, ForeignKey('BotPerformances.Id'), nullable=False)
    TradesPercentToRemove = Column(Float, nullable=False)

    # Relación con BotPerformance original
    BotPerformance = relationship(
        'BotPerformance',
        foreign_keys=[BotPerformanceId],
        back_populates='LuckTest',
        lazy='joined',
        uselist=False
    )

    # Relación con LuckTestPerformance (otro registro en la misma tabla)
    LuckTestPerformance = relationship(
        'BotPerformance',
        foreign_keys=[LuckTestPerformanceId],
        lazy='joined',
        uselist=False
    )

    def __repr__(self):
        return (f"<LuckTest(id={self.Id}, "
                f"BotPerformanceId='{self.BotPerformanceId}', "
                f"LuckTestPerformanceId='{self.LuckTestPerformanceId}', "
                f"TradesPercentToRemove={self.TradesPercentToRemove})>")