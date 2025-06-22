import MetaTrader5 as mt5
import logging

logger = logging.getLogger("TraderBot")

class LivePosition:
    """
    Clase que simula la interfaz de `backtesting.Position` en modo live con MetaTrader 5.
    """

    def __init__(self, strategy):
        self.strategy = strategy
        self.positions = self._get_positions()

    def _get_positions(self):
        """
        Obtiene las posiciones abiertas en MetaTrader 5 para el símbolo del bot.
        """
        positions = mt5.positions_get(symbol=self.strategy.ticker.Name)
        if positions is None:
            return []
        
        return [pos for pos in positions if pos.comment == self.strategy.metatrader_name]

    @property
    def size(self) -> float:
        """Retorna el tamaño total de la posición en lotes (positivo = long, negativo = short)."""
        return sum(pos.volume for pos in self.positions)

    @property
    def pl(self) -> float:
        """Retorna la ganancia o pérdida total en dinero de la posición."""
        return sum(pos.profit for pos in self.positions)

    @property
    def pl_pct(self) -> float:
        """Retorna la ganancia o pérdida total en porcentaje."""
        if not self.positions:
            return 0.0
        
        balance = self.strategy.mt5.account_info().balance
        return (self.pl / balance) * 100

    @property
    def is_long(self) -> bool:
        """Retorna `True` si la posición es long (compra)."""
        return any(pos.type == mt5.ORDER_TYPE_BUY for pos in self.positions)

    @property
    def is_short(self) -> bool:
        """Retorna `True` si la posición es short (venta)."""
        return any(pos.type == mt5.ORDER_TYPE_SELL for pos in self.positions)

    def close(self, portion: float = 1.0):
        """
        Cierra un porcentaje de la posición abierta en MetaTrader 5.
        `portion` debe estar entre 0 y 1.
        """
        if not self.positions:
            return

        for pos in self.positions:
            close_volume = pos.volume * portion
            self.strategy.close_order(pos)

    def __bool__(self):
        """Permite usar `if self.position:` como en `backtesting.py`."""
        return self.size != 0  # Retorna True si hay una posición abierta, False si no

    def __repr__(self):
        return f'<LivePosition: {self.size} ({len(self.positions)} trades)>' 