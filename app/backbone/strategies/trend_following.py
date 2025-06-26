import talib as ta
from app.backbone.strategies.proxy.strategy_factory import StrategyFactory
from backtesting.lib import crossover

class SmaCross(StrategyFactory):
    
    def init(self):
        self.short_sma = self.I(
            ta.SMA, self.data.Close, 12
        )
        
        self.long_sma = self.I(
            ta.SMA, self.data.Close, 50
        )
        
        self.atr = self.I(
            ta.ATR, self.data.High, self.data.Low, self.data.Close,
        )
        
        self.atr_multiplier = 2
        self.risk_reward = 1.5
    
    
    def next(self):
        
        if self.position:
            if self.position.is_long and crossover(self.long_sma, self.short_sma):
                self.position.close()
                
            if self.position.is_short and crossover(self.short_sma, self.long_sma):
                self.position.close()
            
        else:
            price = self.data.Close[-1]
            
            if crossover(self.short_sma, self.long_sma):
                
                sl_price = price - self.atr_multiplier * self.atr[-1]
                tp_price = price + self.risk_reward * self.atr_multiplier * self.atr[-1]

                pip_distance = self.diff_pips(price, sl_price)
                
                units = self.calculate_units_size(
                    price,
                    pip_distance
                )
                
                self.buy(
                    size=units,
                    sl=sl_price,
                    tp=tp_price
                )
                
            if crossover(self.long_sma, self.short_sma):
                sl_price = price + self.atr_multiplier * self.atr[-1]
                tp_price = price - self.risk_reward * self.atr_multiplier * self.atr[-1]

                pip_distance = self.diff_pips(price, sl_price)
                
                units = self.calculate_units_size(
                    price,
                    pip_distance
                )
                
                self.sell(
                    size=units,
                    sl=sl_price,
                    tp=tp_price
                )
            
    