import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.backbone.services.config_service import ConfigService
from app.backbone.services.ticker_service import TickerService
import pandas as pd
import MetaTrader5 as mt5
import yaml
from app.backbone.utils.wfo_utils import (
    optimization_function,
    run_strategy,
)
from app.backbone.utils.general_purpose import load_function
import logging
from pytz import timezone


logger = logging.getLogger("TraderBot")


class BotRunner:

    def __init__(
        self,
        name: str,
        ticker_name: str,
        timeframe: str,
        opt_params: dict,
        wfo_params: dict,
        strategy,
        risk: float,
        timezone,
        data_path: str,
    ):
        self.config_service = ConfigService()
        self.ticker_service = TickerService()

        logger.info(f"inicializando {name}_{ticker_name}_{timeframe}_r{risk}")

        if not mt5.initialize():

            logger.info(f"initialize() failed, error code = {mt5.last_error()}")
            quit()

        name_ticker = ticker_name.split(".")[
            0
        ]  # --> para simbolos como us2000.cash le quito el .cash
        self.metatrader_name = f"{name}_{name_ticker}_{timeframe}"

        if len(self.metatrader_name) > 16:
            raise Exception(
                f"El nombre del bot debe tener un length menor o igual a 16: {self.metatrader_name} tiene {len(self.metatrader_name)}"
            )

        logger.info(
            f"{self.metatrader_name}: Obteniendo apalancamientos para cada activo"
        )
        with open("./app/configs/leverages.yml", "r") as file_name:
            self.leverages = yaml.safe_load(file_name)

        self.mt5 = mt5
        self.ticker = self.ticker_service.get_by_name(name=ticker_name)

        
        self.timeframe = self.ticker_service.get_timeframe_by_name(name=timeframe)

        self.opt_params = opt_params if opt_params != None else {}
        self.wfo_params = wfo_params
        self.strategy = strategy
        self.timezone = timezone

        account_info_dict = self.mt5.account_info()._asdict()
        for prop in account_info_dict:
            logger.info("  {}={}".format(prop, account_info_dict[prop]))

        self.risk = risk
        self.opt_params["maximize"] = optimization_function
        self.data_path = data_path

        logger.info(f"{self.metatrader_name}: Inicializacion completada :)")

    def get_data(self):
        logger.info(f"Intentando leer: {os.path.abspath(self.data_path)}")
        file_path = os.path.join(self.data_path, f"{self.ticker.Name}_{self.timeframe.Name}.csv")

        logger.info(file_path)

        historical_prices = pd.read_csv(file_path)
        historical_prices["Date"] = pd.to_datetime(historical_prices["Date"])
        historical_prices = historical_prices.set_index("Date")
        historical_prices.index = historical_prices.index.tz_localize('UTC').tz_convert('UTC')

        return historical_prices

    def run(self):

        warmup_bars = self.wfo_params["warmup_bars"]
        look_back_bars = self.wfo_params["look_back_bars"]

        logger.info(f"{self.metatrader_name}: Iniciando ejecucion")

        logger.info(f"{self.metatrader_name}: Obteniendo datos historicos")
        prices = self.get_data()

        with open("./app/configs/leverages.yml", "r") as file_name:
            leverages = yaml.safe_load(file_name)

        leverage = leverages[self.ticker.Name]
        margin = 1 / leverage

        initial_cash = float(self.config_service.get_by_name(name='InitialCash').Value)

        _, bt = run_strategy(
            strategy=self.strategy,
            ticker=self.ticker,
            timeframe=self.timeframe,
            prices=prices,
            initial_cash=initial_cash,
            margin=margin,
            risk=self.risk,
            opt_params=None,
            metatrader_name=self.metatrader_name,
            timezone=self.timezone

        )

        bt._results._strategy.change_to_live()
        for indicator in bt._results._strategy._indicators:
            indicator *= bt._results._strategy.minimum_fraction

        bt._results._strategy.next()

        logger.info(f"{self.metatrader_name}: Ejecucion completada")


if __name__ == "__main__":
    strategy_name = "app.backbone.strategies." + sys.argv[1]
    bot_name = sys.argv[2]
    ticker = sys.argv[3]
    timeframe = sys.argv[4]
    risk = float(sys.argv[5])

    opt_params = yaml.safe_load(sys.argv[6]) if sys.argv[6].strip() else {}
    wfo_params = yaml.safe_load(sys.argv[7]) if sys.argv[7].strip() else {}

    metatrader_name = sys.argv[8]
    tz = sys.argv[9]
    data_path = sys.argv[10]

    strategy = load_function(strategy_name)  # Obtiene la clase BPercent

    bot = BotRunner(
        name=metatrader_name,
        ticker_name=ticker,
        timeframe=timeframe,
        opt_params=opt_params,
        wfo_params=wfo_params,
        strategy=strategy,
        risk=risk,
        timezone=timezone(tz),
        data_path=data_path,
    )

    bot.run()