import statistics
from backtesting import Strategy
import MetaTrader5 as mt5
import numpy as np
import telebot
from app.backbone.services.config_service import ConfigService
import logging
from app.backbone.strategies.proxy.live_position import LivePosition
from app.backbone.strategies.proxy.live_trade import LiveTrade

logger = logging.getLogger("TraderBot")


order_tpyes = {
    "buy": mt5.ORDER_TYPE_BUY,
    "sell": mt5.ORDER_TYPE_SELL,
    "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
    "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
}

opposite_order_tpyes = {
    mt5.ORDER_TYPE_BUY: mt5.ORDER_TYPE_SELL,
    mt5.ORDER_TYPE_SELL: mt5.ORDER_TYPE_BUY,
}

def send_telegram_message(message):
    config_service = ConfigService()
    bot_token = config_service.get_by_name(name="TelegramBotToken").Value
    chat_id = config_service.get_by_name(name="TelegramBotChatId").Value

    if bot_token and chat_id:
        bot = telebot.TeleBot(bot_token)
        bot.send_message(chat_id=chat_id, text=message)
        logger.info('Resultado enviado correctamente')

    else:
        logger.info('No se ha configurado un bot_token y chat_id')


class StrategyFactory(Strategy):
    pip_value = None
    minimum_lot = None
    maximum_lot = None
    contract_volume = None
    opt_params = None
    volume_step = None
    risk = None
    metatrader_name = None
    ticker = None
    live = False
    timezone=None
    minimum_fraction=None

    def change_to_live(self):
        self.live = True

    def change_to_bt(self):
        self.live = False

    @property
    def position(self):
        if not self.live:
            return super().position  # Backtesting normal
        
        return LivePosition(self)  # Devuelve proxy en modo live
    
    @property
    def trades(self):
        if not self.live:
            return super().trades  # Backtesting normal
        
        return self._get_positions()
    
    @property
    def equity(self):
        if not self.live:
            return super().equity  # Backtesting normal
        return mt5.account_info().equity  # Equity de MetaTrader

    def calculate_units_size(
        self,
        entry_price,
        stop_loss_pips, 
        verbose=False
        ):
        
        trade_tick_value_loss = None
        categories = ['Forex', 'Metals', 'Crypto', 'Exotics'] # <-- Ajustar

        if self.ticker.Category.Name not in categories or self.ticker.Name.endswith('USD'):
            trade_tick_value_loss = (self.contract_volume * self.pip_value)

        elif self.ticker.Category.Name in categories:

            # El USD no esta en el par
            if 'USD' not in self.ticker.Name:
                conversion_rate = self.data.ConversionRate[-1]
                trade_tick_value_loss = (self.contract_volume * (self.pip_value * conversion_rate))

            # USD es la divisa base ( self.ticker.Name[:3])
            elif self.ticker.Name.startswith('USD'):
                # Crear el valor del dolar como 1 / Open
                trade_tick_value_loss = (self.contract_volume * (self.pip_value / entry_price))


        account_currency_risk = self.equity * (self.risk / 100)
        lots = account_currency_risk / (trade_tick_value_loss * stop_loss_pips)
        
        if verbose:
            logger.info(f'''
                account_currency_risk: {account_currency_risk}, 
                trade_tick_value_loss: {self.trade_tick_value_loss},
                stop_loss_pips: {stop_loss_pips},
                lots: {lots},
            ''')
        
        lots = int(lots * 100) / 100
    
        if self.live: # Si esta en vivo tiene que retornar lotes
            lots = max(lots, self.minimum_lot * self.minimum_fraction)
            lots = min(lots, self.maximum_lot * self.minimum_fraction) 
               
            if self.volume_step < 1:
                number_of_decimals = len(str(self.volume_step).split('.')[1])
                return round(lots, number_of_decimals)
            else:
                return float(int(lots))

        lots = max(lots, self.minimum_lot)
        lots = min(lots, self.maximum_lot)   

        units = int(lots * self.contract_volume)

        return units

    def diff_pips(self, price1, price2, absolute=True):
        if absolute:
            difference = abs(price1 - price2)
        else:
            difference = price1 - price2
        pips = difference / self.pip_value
        
        return pips

    def buy(self, size=None, sl=None, tp=None, limit=None, stop=None):
        """
        Ejecuta una orden de compra en backtesting o en MetaTrader 5.
        """
        if not self.live:
            return super().buy(size=size, sl=sl, tp=tp, limit=limit, stop=stop)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.ticker.Name,
            "volume": size or self.default_lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(self.ticker.Name).ask,
            "magic": 0,
            "comment": self.metatrader_name,
            "type_filling": mt5.ORDER_FILLING_FOK
        }

        if sl: request['sl'] = sl / self.minimum_fraction
        if tp: request['tp'] = tp / self.minimum_fraction
        
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            message = f"{self.metatrader_name}: fallo al abrir orden retcode: {result.retcode}"
            logger.info(message)
            
            try:
                logger.info(f'{self.metatrader_name}: enviando resultado por telegram')
                send_telegram_message(message)
            except:
                logger.exception(f'{self.metatrader_name}: Error al enviar el mensaje por telegram')
                
        else:
            message = f"{self.metatrader_name}: Orden abierta, {result}"
            logger.info(message)
            try:
                logger.info(f'{self.metatrader_name}: enviando resultado por telegram')
                send_telegram_message(message)
                
            except:
                logger.exception(f'{self.metatrader_name}: Error al enviar el mensaje por telegram')

        return result

    def sell(self, size=None, sl=None, tp=None, limit=None, stop=None):
        """
        Ejecuta una orden de venta en backtesting o en MetaTrader 5.
        """
        if not self.live:
            return super().sell(size=size, sl=sl, tp=tp, limit=limit, stop=stop)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.ticker.Name,
            "volume": size or self.default_lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(self.ticker.Name).bid,
            "magic": 0,
            "comment": self.metatrader_name,
            "type_filling": mt5.ORDER_FILLING_FOK
        }

        if sl: request['sl'] = sl / self.minimum_fraction
        if tp: request['tp'] = tp / self.minimum_fraction

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            message = f"{self.metatrader_name}: fallo al abrir orden retcode: {result.retcode}"
            logger.info(message)
            
            try:
                logger.info(f'{self.metatrader_name}: enviando resultado por telegram')
                send_telegram_message(message)
            except:
                logger.exception(f'{self.metatrader_name}: Error al enviar el mensaje por telegram')
                
        else:
            message = f"{self.metatrader_name}: Orden abierta, {result}"
            logger.info(message)
            try:
                logger.info(f'{self.metatrader_name}: enviando resultado por telegram')
                send_telegram_message(message)
                logger.info(f'{self.metatrader_name}: Resultado enviado correctamente')
                
            except:
                logger.exception(f'{self.metatrader_name}: Error al enviar el mensaje por telegram')


        return result
    
    def close_order(self, position):
        
        logger.info(f'{self.metatrader_name}: cerrando posicion {position}')
        
        close_position_type = opposite_order_tpyes[position.type]

        logger.info(f'{self.metatrader_name}: Obtenidendo precio de cierre')

        if close_position_type == order_tpyes["buy"] or order_tpyes["buy_limit"]:
            price = mt5.symbol_info_tick(position.symbol).ask
        else:
            price = mt5.symbol_info_tick(position.symbol).bid
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_position_type,
            "position": position.ticket,
            "price": price,
            "magic": 234000,
            "comment": f"{self.metatrader_name} close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        logger.info(f'{self.metatrader_name}: intendando cerrar orden {request}')
        
        result = mt5.order_send(request)
        
        logger.info(f'{self.metatrader_name}: {result}')

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            message = f"{self.metatrader_name}: fallo al cerrar orden retcode: {result.retcode}"
            logger.info(message)
            
            try:
                logger.info(f'{self.metatrader_name}: enviando resultado por telegram')
                send_telegram_message(message)
            except:
                logger.exception(f'{self.metatrader_name}: Error al enviar el mensaje por telegram')
                
        else:
            message = f"{self.metatrader_name}: Orden cerrada. closed, {result}"
            logger.info(message)
            
            try:
                logger.info(f'{self.metatrader_name}: enviando resultado por telegram')
                send_telegram_message(message)
                
            except:
                logger.exception(f'{self.metatrader_name}: Error al enviar el mensaje por telegram')

    def _get_positions(self):
        """
        Obtiene las posiciones abiertas en MetaTrader 5 para el símbolo del bot.
        """
        trades = mt5.positions_get(symbol=self.ticker.Name)
        if trades is None:
            return []
        
        return [LiveTrade(t, self) for t in trades if t.comment == self.metatrader_name]
    
    def consecutive_bull_candles(self, open_, close, n_candles):
        for i in range(1, n_candles + 1):
            if close[-i] < open_[-i]:
                return False
            
        return True

    def consecutive_bear_candles(self, open_, close, n_candles):
        for i in range(1, n_candles + 1):
            if close[-i] > open_[-i]:
                return False
        
        return True

    def candle_over_the_avg(self, open_, close, window, factor):
        abs_differences = []
        for i in range(2, 2 + window + 1):
            abs_differences.append(
                abs(open_[-i] - close[-i])
            )
            
        mean_difference = statistics.mean(abs_differences)
        actual_difference = abs(close[-1] - open_[-1])
        
        if actual_difference >= factor * mean_difference:
            return True
        
        return False

    def time_in_position(self, trade, max_bars_in_position):
        bars_in_position = len(self.data.index[trade.entry_bar:])

        if bars_in_position > max_bars_in_position:
            return True

        return False
    
    def volume_over_the_avg(self, volume, window, factor):
        mean_volume = np.mean(volume[-(window + 2): -2])
        actual_volume = volume[-1]

        if actual_volume >= factor * mean_volume:
            return True
        
        return False