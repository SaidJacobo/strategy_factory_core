import talib as ta
from app.backbone.strategies.proxy.strategy_factory import StrategyFactory
from backtesting.lib import crossover
import numpy as np
import statistics

class TestStrategy(StrategyFactory):  
    prob_trade = 0.95
    prob_long = 0.5
    prob_short = 0.5
    atr_multiplier = 4
    
    max_bars_in_position = 3
    max_consecutive_candles = 4
    factor = 4
    window = 5

    def init(self):
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close)
        
        if self.opt_params:
            for k, v in self.opt_params.items():
                setattr(self, k, v)
        
        
    def next(self):
    
        if self.position:
            trade = self.trades[-1]
            if self.time_in_position(trade, self.max_bars_in_position):
                self.position.close()
                
            if self.position.is_long:
                bull_candle = (self.data.Close[-1] - self.data.Open[-1]) > 0
                if bull_candle and self.candle_over_the_avg(self.data.Open, self.data.Close, self.window, self.factor):
                    self.position.close()

                if self.consecutive_bull_candles(self.data.Open, self.data.Close, 4):
                    self.position.close()
            
            if self.position.is_short:
                bear_candle = (self.data.Close[-1] - self.data.Open[-1]) < 0
                if bear_candle and self.candle_over_the_avg(self.data.Open, self.data.Close, self.window, self.factor):
                    self.position.close()
                    
                if self.consecutive_bear_candles(self.data.Open, self.data.Close, 4):
                    self.position.close()            


        else:
            trade = None
            long = None
            short = None

            if np.random.rand() < self.prob_trade:
                trade = True
                if np.random.rand() < self.prob_long:
                    long = True
                else:
                    short = True
            
            price = self.data.Close[-1]

            if trade and long:        
                sl_price = self.data.Close[-1] - self.atr_multiplier * self.atr[-1]
                
                pip_distance = self.diff_pips(
                    price, 
                    sl_price, 
                )
                
                units = self.calculate_units_size(
                    price, 
                    stop_loss_pips=pip_distance, 
                )
                
                self.buy(
                    size=units,
                    sl=sl_price
                )

            elif trade and short:        
                sl_price = self.data.Close[-1] + self.atr_multiplier * self.atr[-1]
                
                pip_distance = self.diff_pips(
                    price, 
                    sl_price, 
                )
                
                units = self.calculate_units_size(
                    price, 
                    stop_loss_pips=pip_distance, 
                )
                
                self.sell(
                    size=units,
                    sl=sl_price
                )

                
