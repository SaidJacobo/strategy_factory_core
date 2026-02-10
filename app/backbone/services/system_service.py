import os
from typing import List
from app.backbone.database.db_service import DbService
from app.backbone.entities.grouped_metrics import GroupedMetrics
from app.backbone.entities.portfolio import Portfolio
from app.backbone.entities.system import System
from app.backbone.entities.system_portfolio import SystemPortfolio
from app.backbone.services.config_service import ConfigService
from app.backbone.services.operation_result import OperationResult
from app.backbone.services.portfolio_service import PortfolioService
from app.backbone.services.system_portfolio_metrics_service import SystemPortfolioMetricsService


class SystemService:
    def __init__(self):
        self.db_service = DbService()
        self.config_service = ConfigService()
        self.system_portfolio_metrics_service = SystemPortfolioMetricsService()
        
    def create( self, name:str, description:str) -> OperationResult:
        with self.db_service.get_database() as db:
            
            system_by_filter = self.db_service.get_by_filter(db, System, Name=name)
            
            if system_by_filter is None:
                
                new_system = System(Name=name, Description=description)
                
                system = self.db_service.create(db, new_system)

                result = OperationResult(ok=True, message=None, item=system)
                
                return result
            
            result = OperationResult(ok=False, message='El item ya esta cargado en la BD', item=None)
            return result

    def update(self, id:int, name:str, description:str) -> OperationResult:
        with self.db_service.get_database() as db:
            new_system = System(Id=id, Name=name, Description=description)
            system = self.db_service.update(db, System, new_system)
            
            result = OperationResult(ok=True, message=None, item=system)
            return result

    def delete(self, system_id:int):
        with self.db_service.get_database() as db:
            system = self.db_service.get_by_id(db, System, id=system_id)

            for system_pf in system.SystemPortfolios:
                self.db_service.delete(db, SystemPortfolio, system_pf.Id)

            if system.GroupedMetrics:
                self.db_service.delete(db, GroupedMetrics, system.GroupedMetrics.Id)
            
            self.db_service.delete(db, System, system.Id)
        
        plot_path = f'./app/templates/static/system_plots/{system.Id}.html'
        if os.path.exists(plot_path):
            os.remove(plot_path)

        reports_path = f'./app/templates/static/system_reports/{system.Id}.html'
        if os.path.exists(reports_path):
            os.remove(reports_path)

    def get_all(self) -> List[System]:
        with self.db_service.get_database() as db:
            systems = self.db_service.get_all(db, System)
        
        return systems
    
    def get_by_id(self, system_id:int) -> System:
        with self.db_service.get_database() as db:
            system = self.db_service.get_by_id(db, System, id=system_id)
        
        return system
    
    def get_portfolios_from_system(self, system_id: int) -> List[Portfolio]:
        with self.db_service.get_database() as db:
            system_portfolios = self.db_service.get_many_by_filter(db, SystemPortfolio, SystemId=system_id)
            
            portfolios = [systems_pf.Portfolio for systems_pf in system_portfolios ]
            
            return portfolios
        
    def get_system_portfolios(self, system_id:int) -> List[SystemPortfolio]:
        with self.db_service.get_database() as db:
            system_portfolios = self.db_service.get_many_by_filter(db, SystemPortfolio, SystemId=system_id)
            
            return system_portfolios

    def get_system_portfolio(self, system_id: int, portfolio_id:int) -> SystemPortfolio:
        with self.db_service.get_database() as db:
            system_portfolio = self.db_service.get_by_filter(
                db, SystemPortfolio, SystemId=system_id, PortfolioId=portfolio_id
            )
            
            return system_portfolio

    def add_portfolio(self, system_id: int, portfolio_id: int) -> OperationResult:
        # Chequeo previo: ¿ya existe esa relación?
        result = self.get_system_portfolio(
            system_id=system_id,
            portfolio_id=portfolio_id,
        )
        
        if result:
            return OperationResult(
                ok=False,
                message='This portfolio is already in the system',
                item=None
            )

        # Abrimos una única sesión para todo el flujo
        with self.db_service.get_database() as db:
            # Cargamos las entidades dentro de esta sesión
            system = db.query(System).get(system_id)
            portfolio = db.query(Portfolio).get(portfolio_id)

            if not system:
                return OperationResult(
                    ok=False,
                    message='System not found',
                    item=None
                )
            
            if not portfolio:
                return OperationResult(
                    ok=False,
                    message='Portfolio not found',
                    item=None
                )

            # Creamos la relación intermedia
            system_portfolio = SystemPortfolio(
                SystemId=system.Id,
                PortfolioId=portfolio.Id,
                System=system,
                Portfolio=portfolio
            )

            _ = self.db_service.create(db, system_portfolio)

            return OperationResult(ok=True, message=None, item=None)
        
    def delete_portfolio(self, system_id: int, portfolio_id:int) -> OperationResult:
        system_portfolio_result = self.get_system_portfolio(
            system_id=system_id,
            portfolio_id=portfolio_id,
        )
        
                
        if not system_portfolio_result:
            return OperationResult(ok=False, message='This portfolio is not in the system', item=None)
        
        with self.db_service.get_database() as db:
            self.db_service.delete(db, SystemPortfolio, system_portfolio_result.Id)
            return OperationResult(ok=True, message=None, item=None)
        
    def set_portfolio_weights(self, system_id: int, portfolio_ids: int | List[int], weights: float | List[float]):
        
        portfolio_ids = [portfolio_ids] if not type(portfolio_ids) == list else portfolio_ids
        weights = [weights] if not type(weights) == list else weights

        message = None
        with self.db_service.get_database() as db:
            system_portfolios = self.db_service.get_many_by_filter(db, SystemPortfolio, SystemId=system_id)

            for portfolio_id, weight in zip(portfolio_ids, weights):
                
                portfolio = [sys_pf for sys_pf in system_portfolios if sys_pf.PortfolioId == portfolio_id].pop()

                if not portfolio:
                    message = '' if not message else message
                    message += f'portfolio_id {portfolio_id} is not in system {system_id}'
                
                portfolio.Weight = weight
                self.db_service.update(db, SystemPortfolio, portfolio)

        return OperationResult(ok=True, message=message, item=None)