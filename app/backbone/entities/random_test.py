from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from . import Base

class RandomTest(Base):
    __tablename__ = 'RandomTests'

    Id = Column(Integer, primary_key=True, autoincrement=True)
    BotPerformanceId = Column(Integer, ForeignKey('BotPerformances.Id'), nullable=False)
    RandomTestPerformanceId = Column(Integer, ForeignKey('BotPerformances.Id'), nullable=True)
    Iterations = Column(Integer, nullable=False)
    
    ReturnDdMeanDiff = Column(Float, nullable=True)
    ReturnDdStdDiff = Column(Float, nullable=True)
    ReturnDdPValue = Column(Float, nullable=True)
    ReturnDdZScore = Column(Float, nullable=True)

    ReturnMeanDiff = Column(Float, nullable=True)
    ReturnStdDiff = Column(Float, nullable=True)
    ReturnPValue = Column(Float, nullable=True)
    ReturnZScore = Column(Float, nullable=True)

    DrawdownMeanDiff = Column(Float, nullable=True)
    DrawdownStdDiff = Column(Float, nullable=True)
    DrawdownPValue = Column(Float, nullable=True)
    DrawdownZScore = Column(Float, nullable=True)

    WinrateMeanDiff = Column(Float, nullable=True)
    WinrateStdDiff = Column(Float, nullable=True)
    WinratePValue = Column(Float, nullable=True)
    WinrateZScore = Column(Float, nullable=True)

    # Relación con BotPerformance original
    BotPerformance = relationship(
        'BotPerformance',
        foreign_keys=[BotPerformanceId],
        back_populates='RandomTest',
        lazy='joined',
        uselist=False
    )

    RandomTestPerformance = relationship(
        'BotPerformance',
        foreign_keys=[RandomTestPerformanceId],
        lazy='joined',
        uselist=False
    )

    def __repr__(self):
        return (f"<RandomTest(id={self.Id}, "
                f"BotPerformanceId='{self.BotPerformanceId}', "
                f"RandomTestPerformanceId='{self.RandomTestPerformanceId}', "
                f"Iterations={self.Iterations})>")