import asyncio
import json
import os
from typing import List
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pandas import Timestamp
import yaml
from app.backbone.services.backtest_service import BacktestService
from app.backbone.services.config_service import ConfigService
from app.backbone.services.portfolio_service import PortfolioService
from app.backbone.services.system_portfolio_metrics_service import SystemPortfolioMetricsService
from app.backbone.services.system_service import SystemService
from app.backbone.utils.general_purpose import build_live_trading_config, save_ticker_timeframes
from app.backbone.utils.logger import get_logger
from app.view_models.op_result_vm import OperationResultVM
from app.view_models.portfolio_vm import PortfolioVM
from app.view_models.system_portfolio_vm import SystemPortfolioVM
from app.view_models.system_vm import SystemVM
from app.view_models.update_system_portfolio import UpdateSystemPortfolioVM
import pandas as pd
import quantstats as qs


logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="./app/templates")
system_service = SystemService()
portfolio_service = PortfolioService()
config_service = ConfigService()
backtest_service = BacktestService()
system_portfolio_metrics_service = SystemPortfolioMetricsService()

@router.get("/systems", response_class=HTMLResponse)
async def systems_index(request: Request):
    try:
        systems = system_service.get_all()
        systems_vm = [SystemVM.model_validate(system) for system in systems]

        return templates.TemplateResponse("/systems/index.html", {"request": request, "systems": systems_vm})
    
    except Exception as e:
        logger.info(f'There was an error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get("/systems/new", response_class=HTMLResponse)
async def create_get(request: Request):
    try:
        return templates.TemplateResponse("/systems/create.html", {"request": request})
    
    except Exception as e:
        logger.info(f'There was an error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/systems")
async def create_post(
    request:Request,
    name: str = Form(...),
    description: str = Form(...),
):
    try:
        result = system_service.create(name=name, description=description)
        
        if result.ok:
            return RedirectResponse(url="/systems", status_code=303)

        else:
            logger.info(f'There was an error {str(e)}')
            return templates.TemplateResponse("/error.html", {"request": request, 'error': result.message})

    except Exception as e:
        logger.info(f'There was an error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/systems/{system_id}/delete")
async def delete_system(request: Request, system_id: int):
    try:
        _ = system_service.delete(system_id=system_id)
        return RedirectResponse(url="/systems/", status_code=303)
    
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get("/systems/{system_id}")
async def update_get(request: Request, system_id: int):

    try:
        system = system_service.get_by_id(system_id)
        system_vm = SystemVM.model_validate(system)

        return templates.TemplateResponse("/systems/update.html", {"request": request, "system": system_vm})
        
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.post("/systems/{system_id}")
async def update_post(
    request:Request,
    id:int = Form(...),
    name: str = Form(...),
    description: str = Form(...),
):
    try:
        _ = system_service.update(id=id, name=name, description=description)
        return RedirectResponse(url="/systems", status_code=303)
        
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})
 
@router.get("/systems/admin/{system_id}", response_class=HTMLResponse)
async def get_systems_admin(request: Request, system_id:int):
    
    try:
        system = system_service.get_by_id(system_id=system_id)
        
        system_vm = SystemVM.model_validate(system)

        # Obtengo todos los backtest de un portfolio
        used_portfolios = system_service.get_portfolios_from_system(system_id=system_id)
        used_portfolios = [PortfolioVM.model_validate(portfolio) for portfolio in used_portfolios]

        system_vm.Portfolios = used_portfolios

        system_equity_plot = None
        system_plot_path = './app/templates/static/system_plots'
        file_name = f'{system.Id}.html'
        if os.path.exists(os.path.join(system_plot_path, file_name)):
            with open(os.path.join(system_plot_path, file_name), 'r') as f:
                system_equity_plot = json.load(f)  # Cargar el contenido JSON
            system_vm.HasEquityPlot = True

        return templates.TemplateResponse(
            "/systems/admin.html", 
            {
                "request": request, 
                'system':system_vm,
                'equity_plot': system_equity_plot or {"data": [], "layout": {}},

            }
        )

    except Exception as e:
        logger.info(f'There was an error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get("/systems/admin/{system_id}/candidates", response_class=HTMLResponse)
def get_candidates(request:Request, system_id:int):
    try:
        # Obtener todos los bt robustos
        used_portfolios = system_service.get_portfolios_from_system(system_id=system_id)
        used_portfolios_ids = [pf.Id for pf in used_portfolios]

        portfolios = portfolio_service.get_all()
        portfolios_vm = [PortfolioVM.model_validate(pf) for pf in portfolios if pf.Id not in used_portfolios_ids]
        
        return templates.TemplateResponse("/systems/modal_candidates.html", {
            "request": request, 
            'portfolios': portfolios_vm,
            'system_id': system_id
        })
    
    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})


@router.post("/systems/admin/{system_id}/add/{portfolio_id}")
async def add_bt_portfolios(system_id:int, portfolio_id:int):

    try:    
        result = system_service.add_portfolio(system_id=system_id, portfolio_id=portfolio_id)
        result = OperationResultVM.model_validate(result)
        return JSONResponse(result.model_dump_json())
    
    except Exception as e:
        logger.info(f'There was an error: {str(e)}')
        result = OperationResultVM(ok=False, message='There was an error to add the portfolio in the system.', item=None)
        return JSONResponse(result.model_dump_json())


@router.post("/systems/admin/{system_id}/delete/{portfolio_id}")
async def delete_bt_portfolios(system_id:int, portfolio_id:int):
    try:
        result = system_service.delete_portfolio(system_id=system_id, portfolio_id=portfolio_id)
        result = OperationResultVM.model_validate(result)
        return JSONResponse(result.model_dump_json())

    except Exception as e:
        logger.info(f'There was an error: {str(e)}')
        result = OperationResultVM(ok=False, message='There was an error to delete the porfolio', item=None)
        return JSONResponse(result.model_dump_json())


@router.get("/systems/admin/{system_id}/correlations", response_class=HTMLResponse)
async def get_system_correlations(request: Request, system_id:int):
    try:

        portfolios = system_service.get_portfolios_from_system(system_id=system_id)

        portfolio_equity_curves = {}
        for portfolio in portfolios:

            trades_with_equity = portfolio_service.get_df_trades(portfolio_id=portfolio.Id)
            portfolio_equity_curve = system_portfolio_metrics_service.get_full_equity_curve(trades_with_equity)
            
            portfolio_equity_curves[portfolio.Name] = portfolio_equity_curve

        initial_portfolio_cash = config_service.get_by_name(name='InitialCash').Value
        correlation_plot = system_portfolio_metrics_service.get_portfolio_correlation_matrix_plot(portfolio_equity_curves, initial_cash=float(initial_portfolio_cash))

        return templates.TemplateResponse(
            "/portfolios/modal_correlations.html", 
            {
                "request": request, 
                'correlation_plot': correlation_plot
            }
        )

    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get("/systems/admin/{system_id}/weights", response_class=HTMLResponse)
def get_weights(request:Request, system_id:int):
    try:
        
        system_portfolios = system_service.get_system_portfolios(system_id=system_id)

        system_portfolios_vm = []
        for sys_pf  in system_portfolios:
            system_portfolio_vm = SystemPortfolioVM(
                PortfolioId=sys_pf.PortfolioId,
                PortfolioName=sys_pf.Portfolio.Name, 
                PortfolioWeight= 0 if not sys_pf.Weight else sys_pf.Weight
            )

            system_portfolios_vm.append(system_portfolio_vm)

        update_system_portfolio_vm = UpdateSystemPortfolioVM(
            SystemId=system_id, SystemWeights=system_portfolios_vm
        )     

        return templates.TemplateResponse("/systems/modal_weights.html", {
            "request": request,
            'vm': update_system_portfolio_vm
        })
    
    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.post("/systems/admin/{system_id}/weights")
def set_weights(system_id: int, system_portfolios: List[SystemPortfolioVM]):
    try:
        op_result = system_service.set_portfolio_weights(
            system_id=system_id,
            portfolio_ids=[sys_pf.PortfolioId for sys_pf in system_portfolios],
            weights=[pf_bt.PortfolioWeight for pf_bt in system_portfolios],
        )

        op_result = OperationResultVM.model_validate(op_result)

        return JSONResponse(content=op_result.model_dump())     

    except Exception as e:
        logger.info(f'Error {str(e)}')
        result = OperationResultVM(ok=False, message='There was an error..', item=None)
        return JSONResponse(content=result.model_dump())
    

@router.post("/systems/run/{system_id}")
async def run_system(
    request: Request, 
    system_id: int, 
    date_from:str=Form(default=None), 
    date_to:str=Form(default=None), 
):
    try:
        system = system_service.get_by_id(system_id=system_id)

        date_from = Timestamp(date_from, tz="UTC") if date_from else None
        date_to = Timestamp(date_to, tz="UTC") if date_to else None

        system_portfolios = system_service.get_system_portfolios(system_id=system_id)
        queue = asyncio.Queue()
        
        all_trades = {}
        equity_curves = {}

        # Iteracion sobre todos los portfolios vinculados al sistema
        for sys_pf  in system_portfolios:
            portfolio_trades = {}
            portfolio = sys_pf.Portfolio
            portfolio_backtests = portfolio_service.get_portfolio_backtests(portfolio_id=sys_pf.PortfolioId)
            initial_cash = float(config_service.get_by_name('InitialCash').Value)

            # Iteracion por cada backtest del portfolio
            for pf_bt in portfolio_backtests:
                date_from = date_from if date_from else Timestamp(pf_bt.BotPerformance.DateFrom, tz="UTC")
                date_to = date_to if date_to else Timestamp(pf_bt.BotPerformance.DateTo, tz="UTC")
                
                risk = round(sys_pf.Weight * pf_bt.Weight, 3)
                _, _, stats = await backtest_service.run_backtest(
                    initial_cash, 
                    pf_bt.BotPerformance.Bot.Strategy, 
                    pf_bt.BotPerformance.Bot.Ticker, 
                    pf_bt.BotPerformance.Bot.Timeframe,
                    date_from, 
                    date_to, 
                    'pa', 
                    risk, 
                    save_bt_plot='discard', 
                    queue=queue,
                )

                trades = stats._trades
                trades['Id'] = trades.index
                trades['ExitTime'] = pd.to_datetime(trades['ExitTime'])
                trades = trades.sort_values(by='ExitTime')
                trades['Date'] = trades['ExitTime']
                trades.set_index('Date', inplace=True)

                portfolio_trades[pf_bt.BotPerformance.Bot.Name] = trades
                all_trades[pf_bt.BotPerformance.Bot.Name] = trades
            

            portfolio_equity_curve = system_portfolio_metrics_service.get_full_equity_curve(portfolio_trades)
            equity_curves[portfolio.Name] = portfolio_equity_curve
            
        system_equity_curve = system_portfolio_metrics_service.get_full_equity_curve(all_trades)
        equity_curves[system.Name] = system_equity_curve

        system_portfolio_metrics_service.calculate_metrics_and_save(
            system_id=system.Id, 
            equity_curve=system_equity_curve, 
            all_trades=all_trades
        )

        equity_plot = system_portfolio_metrics_service.get_full_equity_curve_plot(equity_curves)

        plot_path = './app/templates/static/system_plots'
        file_name = f'{system_id}.html'
        with open(os.path.join(plot_path, file_name), 'w') as f:
            f.write(equity_plot)

        returns = system_equity_curve.Equity.pct_change().dropna()

        plot_path = './app/templates/static/system_reports'
        file_name = f'{system.Id}.html'
        full_path = os.path.join(plot_path, file_name)

        risk_free_rate = float(config_service.get_by_name('RiskFreeRate').Value)

        qs.reports.html(
            returns, 
            output=full_path, 
            title=system.Name, 
            rf=risk_free_rate
        )

        return RedirectResponse(f"/systems/admin/{system_id}", status_code=303)

    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})


@router.post("/systems/admin/{system_id}/deploy")
async def deploy_system(request: Request, system_id: int):
    system_portfolios = system_service.get_system_portfolios(system_id=system_id)
    
    #risk = round(sys_pf.Weight * pf_bt.Weight, 3)
    
    all_backtests = []
    for sys_pf in system_portfolios:
        portfolio_backtests = portfolio_service.get_portfolio_backtests(portfolio_id=sys_pf.PortfolioId)
        for pf_bt in portfolio_backtests:
            # Inyectamos el peso del portafolio en cada backtest usando una propiedad temporal
            bot_performance = pf_bt.BotPerformance
            bot_performance._weight = round(sys_pf.Weight * pf_bt.Weight, 3)
            all_backtests.append(bot_performance)

    def risk_with_weight(bp): return bp._weight

    config_file = build_live_trading_config(all_backtests, risk_with_weight)

    with open("./app/configs/live_trading_auto.yml", "w") as file:
        yaml.dump(config_file, file, default_flow_style=False, allow_unicode=True)
    
    ticker_timeframes = save_ticker_timeframes(config_file)

    with open("./app/configs/metatrader.yml", "w") as file:
        yaml.dump(
            ticker_timeframes, 
            file, 
            default_flow_style=False, 
            allow_unicode=True
        )

    return {'ok': True, 'message': None}
