import MetaTrader5 as mt5
from datetime import datetime
from typing import Optional

import pandas as pd

class LiveTrade:
    """
    Clase que replica la interfaz de `backtesting.Trade` pero con datos en vivo desde MetaTrader 5.
    """

    def __init__(self, deal, strategy):
        self.strategy = strategy
        self.deal = deal  # objeto de tipo _asdict() de mt5.history_deals_get
        self.ticket = deal.ticket
        self._closed = False
        self.exit_deal = None

    @property
    def size(self):
        return self.deal.volume * (1 if self.deal.type == mt5.ORDER_TYPE_BUY else -1)

    @property
    def entry_price(self):
        return self.deal.price

    @property
    def exit_price(self) -> Optional[float]:
        return self.exit_deal.price if self.exit_deal else None

    @property
    def entry_time(self):
        return pd.Timestamp(self.deal.time, unit='s', tz=self.strategy.timezone)

    @property
    def exit_time(self) -> Optional[pd.Timestamp]:
        return pd.Timestamp(self.exit_deal.time, unit='s', tz=self.strategy.timezone) if self.exit_deal else None
        @property
        def is_long(self):
            return self.size > 0

    @property
    def is_short(self):
        return self.size < 0

    @property
    def pl(self):
        if self.exit_deal:
            return self.exit_deal.profit
        else:
            return self._current_unrealized_pl()

    def _current_unrealized_pl(self):
        symbol_info = mt5.symbol_info_tick(self.strategy.ticker)
        if not symbol_info:
            return 0.0
        price = symbol_info.bid if self.is_long else symbol_info.ask
        diff = price - self.entry_price if self.is_long else self.entry_price - price
        return diff * abs(self.size) * self.strategy.lot_value

    @property
    def pl_pct(self):
        cost = abs(self.entry_price * self.size * self.strategy.lot_value)
        return (self.pl / cost) * 100 if cost != 0 else 0

    @property
    def value(self):
        symbol_info = mt5.symbol_info_tick(self.strategy.ticker)
        if not symbol_info:
            return 0.0
        price = symbol_info.bid if self.is_long else symbol_info.ask
        return abs(self.size) * price * self.strategy.lot_value

    @property
    def commission(self):
        return self.deal.commission

    def close(self, portion: float = 1.0):
        self.strategy.close_trade(self, portion)
        self._closed = True

    def __repr__(self):
        return f"<LiveTrade size={self.size} entry={self.entry_price} pl={self.pl:.2f}>"

    @property
    def entry_bar(self):
        return self._get_bar_index(self.entry_time)

    @property
    def exit_bar(self):
        if self.exit_time:
            return self._get_bar_index(self.exit_time)
        else:
            return -1  # La posición sigue abierta

    def _get_bar_index(self, timestamp: pd.Timestamp):
        # Asegura misma zona horaria
        index_tz = self.strategy.data.index.tz
        ts = timestamp.tz_convert(index_tz) if index_tz else timestamp.tz_localize(None)

        # Redondea al inicio de la vela (ej: minuto)
        ts = ts.floor(freq=self.strategy.data.index.freq or 'T')

        # Busca el índice más cercano
        idx = self.strategy.data.index.get_indexer([ts], method='bfill')[0]
        return idx if idx >= 0 else -1