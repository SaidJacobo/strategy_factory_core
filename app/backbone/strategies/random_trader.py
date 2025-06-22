import talib as ta
import numpy as np

from app.backbone.strategies.proxy.strategy_factory import StrategyFactory

class RandomTrader(StrategyFactory):  
    prob_trade = 0.5
    prob_long = 0.5
    prob_short = 0.5
    avg_position_hold = 2
    std_position_hold = 1.5
    
    pos_hold = 0
    max_pos_hold = 0
    
    atr_multiplier = 1.5

    def init(self):
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close)
        
        if self.opt_params:
            for k, v in self.opt_params.items():
                setattr(self, k, v)
        
        
    def next(self):
    
        if self.position:
            self.pos_hold += 1
            
            if self.pos_hold >= self.max_pos_hold:
                self.position.close()
                self.pos_hold = 0

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
                
                self.max_pos_hold = np.round(np.random.normal(self.avg_position_hold, self.std_position_hold, 1)[0])
                
            if trade and short:        
                sl_price = price + self.atr_multiplier * self.atr[-1]
                
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
                
                self.max_pos_hold = np.round(np.random.normal(self.avg_position_hold, self.std_position_hold, 1)[0])
                
                
    