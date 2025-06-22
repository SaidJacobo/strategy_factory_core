from typing import List
from app.backbone.entities.bot import Bot
from app.backbone.entities.timeframe import Timeframe
from app.backbone.database.db_service import DbService
from app.backbone.services.operation_result import OperationResult
from app.backbone.entities.ticker import Ticker
from app.backbone.entities.category import Category
import MetaTrader5 as mt5

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

class TickerService:
    def __init__(self):
        self.db_service = DbService()


    def create_update_timeframes(self) -> OperationResult:
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

        with self.db_service.get_database() as db:
            # Guardar en la base de datos si no existe
            for name, number in TIMEFRAMES.items():
                result_timeframe = self.db_service.get_by_filter(db, Timeframe, Name=name)

                if not result_timeframe:
                    timeframe = Timeframe(
                        Name=name,
                        MetaTraderNumber=number,
                        Selected=False,
                    )

                    self.db_service.create(db, timeframe)
                
                else:
                    timeframe = result_timeframe
                    timeframe.Selected = False if not timeframe.Selected else True
                    self.db_service.update(db, Timeframe, timeframe)


            # Confirmar los cambios en la base de datos
            self.db_service.save(db)

        return OperationResult(ok=True, message="Categorías y tickers procesados correctamente", item=None)

    def update_timeframe(self, timeframe: Timeframe, selected:bool):
        with self.db_service.get_database() as db:

            timeframe.Selected = selected

            self.db_service.update(db, Timeframe, timeframe)
            self.db_service.save(db)
        
    def create(self) -> OperationResult:
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

        symbols = mt5.symbols_get()

        categories_tickers = {}
        for symbol in symbols:
            category_name = symbol.path.split("\\")[0]
            ticker_name = symbol.path.split("\\")[1]

            if category_name not in categories_tickers.keys():
                categories_tickers[category_name] = []

            categories_tickers[category_name].append(ticker_name)

        with self.db_service.get_database() as db:
            for category_name, tickers in categories_tickers.items():
                # Buscar la categoría en la base de datos
                category = self.db_service.get_by_filter(db, Category, Name=category_name)
                
                # Si la categoría no existe, crearla
                if not category:
                    category = Category(Name=category_name, Commission=0)
                    self.db_service.create(db, category)

                # Procesar los tickers asociados a esta categoría
                for ticker_name in tickers:
                    print(ticker_name)

                    _ = mt5.copy_rates_from_pos(
                        ticker_name, 16385, 0, 3
                    )

                    symbol_info = mt5.symbol_info(ticker_name)

                    if symbol_info is not None:
                        print(symbol_info)

                        while symbol_info.ask == 0:
                            symbol_info = mt5.symbol_info(ticker_name)

                        spread = (symbol_info.spread * symbol_info.point) / symbol_info.ask

                        # Buscar el ticker en la base de datos
                        ticker = self.db_service.get_by_filter(db, Ticker, Name=ticker_name, CategoryId=category.Id)
                        
                        # Si el ticker no existe, crearlo
                        if not ticker:
                            ticker = Ticker(Name=ticker_name, Category=category, Spread=spread)
                            self.db_service.create(db, ticker)
                        else:
                            # Si el ticker existe, actualizar su información
                            ticker.Spread = spread
                            self.db_service.update(db, Ticker, ticker)

            self.db_service.save(db)

        return OperationResult(ok=True, message="Categorías y tickers procesados correctamente", item=None)

    def get_tickers_by_category(self, category_id:int) -> List[Ticker]:
        with self.db_service.get_database() as db:
            tickers = self.db_service.get_many_by_filter(db, Ticker, CategoryId=category_id)
            
            return tickers
        
    def get_by_name(self, name:str) -> Ticker:
        with self.db_service.get_database() as db:
            ticker = self.db_service.get_by_filter(db, Ticker, Name=name)
            
        return ticker

    def get_all_categories(self) -> List[Category]:
        with self.db_service.get_database() as db:
            categories = self.db_service.get_all(db, Category)
            return categories
            
    def get_ticker_by_id(self, id) -> Ticker:
        with self.db_service.get_database() as db:
            ticker = self.db_service.get_by_id(db, Ticker, id)
            return ticker
            
    def get_all_timeframes(self) -> List[Timeframe]:
        with self.db_service.get_database() as db:
            categories = self.db_service.get_all(db, Timeframe)
            return categories

    def get_timeframe_by_id(self, id) -> Timeframe:
        with self.db_service.get_database() as db:
            timeframe = self.db_service.get_by_id(db, Timeframe, id)
            return timeframe
        
    def get_timeframe_by_name(self, name:str) -> Timeframe:
        with self.db_service.get_database() as db:
            timeframe = self.db_service.get_by_filter(db, Timeframe, Name=name)
            return timeframe
                        
    def get_all_tickers(self) -> List[Ticker]:
        with self.db_service.get_database() as db:
            tickers = self.db_service.get_all(db, Ticker)
            return tickers
         
    def get_tickers_by_strategy(self, strategy_id) -> OperationResult:
        with self.db_service.get_database() as db:
        
            strategies = (
                db.query(Ticker)
                    .join(Bot, Bot.TickerId == Ticker.Id)
                    .filter(Bot.StrategyId == strategy_id)
            )
            
            result = OperationResult(ok=True, message=None, item=strategies)
            return result

    def update_categories_commissions(self, category_id: int | List[int], commission:float | List[float]):

        if type(category_id) != list:
            category_id = [category_id]

        if type(commission) != list:
            commission = [commission]

        if len(commission) != len(category_id):
            return OperationResult(ok=False, message='The length of the lists must be the same')

        with self.db_service.get_database() as db:
            for cat_id, com in zip(category_id, commission):
                old_category = self.db_service.get_by_id(db, Category, id=cat_id)
                new_category = old_category
                new_category.Commission = com

                self.db_service.update(db, Category, new_category)

            return OperationResult(ok=True, message=None, item=None)

        