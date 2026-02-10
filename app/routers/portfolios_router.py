import asyncio
import json
import os
from typing import List
from pandas import Timestamp
from app.backbone.services.config_service import ConfigService
from app.backbone.services.system_portfolio_metrics_service import SystemPortfolioMetricsService
from app.backbone.utils.general_purpose import build_live_trading_config, save_ticker_timeframes
from app.view_models.op_result_vm import OperationResultVM
from fastapi import APIRouter, HTTPException
from fastapi import Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.backbone.services.backtest_service import BacktestService
from app.backbone.services.portfolio_service import PortfolioService
from app.view_models.performance_metrics_vm import PerformanceMetricsVM
from app.view_models.portfolio_backtest_vm import PortfolioBacktestVM
from app.view_models.grouped_metrics_vm import GroupedMetricsVM
from app.view_models.portfolio_vm import PortfolioVM
import yaml
from app.backbone.utils.logger import get_logger
import quantstats as qs
from app.view_models.update_portfolio_backtest_vm import UpdatePortfolioBacktestVM

logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="./app/templates")
portfolio_service = PortfolioService()
config_service = ConfigService()
backtest_service = BacktestService()
system_portfolio_metrics_service = SystemPortfolioMetricsService()

# Ruta GET: muestra el formulario
@router.get("/portfolios", response_class=HTMLResponse)
async def form_page(request: Request):
    try:
        portfolios = portfolio_service.get_all()
        portfoliosvm = [PortfolioVM.model_validate(portfolio) for portfolio in portfolios]

        return templates.TemplateResponse("/portfolios/index.html", {"request": request, "portfolios": portfoliosvm})
    
    except Exception as e:
        logger.info(f'There was an error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get("/portfolios/new", response_class=HTMLResponse)
async def create_get(request: Request):
    try:
        return templates.TemplateResponse("/portfolios/create.html", {"request": request})
    
    except Exception as e:
        logger.info(f'There was an error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/portfolios")
async def create_post(
    request:Request,
    name: str = Form(...),
    description: str = Form(...),
):
    try:
        result = portfolio_service.create(name=name, description=description)
        
        if result.ok:
            return RedirectResponse(url="/portfolios", status_code=303)

        else:
            logger.info(f'Hubo un error cargando el portfolio {str(e)}')
            return templates.TemplateResponse("/error.html", {"request": request, 'error': result.message})

    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/portfolios/{portfolio_id}/delete")
async def delete_portfolio(request: Request, portfolio_id: int):
    try:
        _ = portfolio_service.delete(portfolio_id=portfolio_id)
        return RedirectResponse(url="/portfolios/", status_code=303)
    
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get("/portfolios/{portfolio_id}/candidates", response_class=HTMLResponse)
def get_candidates(request:Request, portfolio_id:int):
    try:
        # Obtener todos los bt robustos
        robusts_backtests = backtest_service.get_robusts()
        
        # OBtener todos los favoritos
        favorites_backtests = backtest_service.get_favorites()
        
        used_backtests = portfolio_service.get_backtests_from_portfolio(portfolio_id=portfolio_id)
        used_backtests_ids = [bt.Id for bt in used_backtests]

        unique_backtests = {bt.Id: bt for bt in robusts_backtests + favorites_backtests if bt.Id not in used_backtests_ids}
        
        favorites_and_robusts = list(unique_backtests.values())
        favorites_and_robusts = [PerformanceMetricsVM.model_validate(backtest) for backtest in favorites_and_robusts]
        
        return templates.TemplateResponse("/portfolios/modal_candidates.html", {
            "request": request, 
            'favorites_and_robusts': favorites_and_robusts,
            'portfolio_id': portfolio_id
        })
    
    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get("/portfolios/admin/{portfolio_id}/correlations", response_class=HTMLResponse)
async def get_portfolio_correlations(request: Request, portfolio_id:int):
    try:
        portfolio = portfolio_service.get_by_id(portfolio_id=portfolio_id)
        if not portfolio:
            pass
        # Obtengo todos los backtest de un portfolio
        used_backtests = portfolio_service.get_backtests_from_portfolio(portfolio_id=portfolio_id)
        
        if not used_backtests:
            return templates.TemplateResponse(
            "/portfolios/portfolio_metrics.html", 
            {
                "request": request, 
                'metrics': None,
                'equity_plot': None
            }
        )
        
        trades_with_equity = portfolio_service.get_df_trades(portfolio_id)

        initial_portfolio_cash = config_service.get_by_name(name='InitialCash').Value

        correlation_plot = system_portfolio_metrics_service.get_portfolio_correlation_matrix_plot(trades_with_equity, initial_cash=float(initial_portfolio_cash))

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

@router.get("/portfolios/admin/{portfolio_id}", response_class=HTMLResponse)
async def get_portfolios_admin(request: Request, portfolio_id:int):
    
    try:
        portfolio = portfolio_service.get_by_id(portfolio_id=portfolio_id)
        
        # Obtengo todos los backtest de un portfolio
        used_backtests = portfolio_service.get_backtests_from_portfolio(portfolio_id=portfolio_id)
        used_backtests = [PerformanceMetricsVM.model_validate(backtest) for backtest in used_backtests]
        
        portfolio_vm = PortfolioVM.model_validate(portfolio)
        portfolio_vm.BotPerformances = used_backtests

        portfolio_equity_plot = None
        portfolio_plot_path = './app/templates/static/portfolio_plots'
        file_name = f'{portfolio_id}.html'
        if os.path.exists(os.path.join(portfolio_plot_path, file_name)):
            with open(os.path.join(portfolio_plot_path, file_name), 'r') as f:
                portfolio_equity_plot = json.load(f)  # Cargar el contenido JSON
            portfolio_vm.HasEquityPlot = True
        
        return templates.TemplateResponse(
            "/portfolios/admin.html", 
            {
                "request": request, 
                'portfolio':portfolio_vm,
                'equity_plot': portfolio_equity_plot or {"data": [], "layout": {}},
            }
        )

    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/portfolios/admin/{portfolio_id}/add/{bot_performance_id}")
async def add_bt_portfolios(portfolio_id:int, bot_performance_id:int):

    try:    
        result = portfolio_service.add_performance(portfolio_id=portfolio_id, bot_performance_id=bot_performance_id)
        result = OperationResultVM.model_validate(result)
        return JSONResponse(result.model_dump_json())
    
    except Exception as e:
        logger.info(f'Error al agregar backtest al portfolio: {str(e)}')
        result = OperationResultVM(ok=False, message='There was an error to add the bot in the porfolio.', item=None)
        return JSONResponse(result.model_dump_json())

@router.post("/portfolios/admin/{portfolio_id}/delete/{bot_performance_id}")
async def delete_bt_portfolios(portfolio_id:int, bot_performance_id:int):
    try:
        result = portfolio_service.delete_performance(portfolio_id=portfolio_id, bot_performance_id=bot_performance_id)
        result = OperationResultVM.model_validate(result)
        return JSONResponse(result.model_dump_json())

    except Exception as e:
        logger.info(f'Error al eliminar backtest al portfolio: {str(e)}')
        result = OperationResultVM(ok=False, message='There was an error to delete the bot in the porfolio.', item=None)
        return JSONResponse(result.model_dump_json())

@router.post("/portfolios/admin/{portfolio_id}/deploy")
async def deploy_portfolio(request: Request, portfolio_id: int):
    used_backtests = portfolio_service.get_backtests_from_portfolio(portfolio_id=portfolio_id)

    def risk_plain(bt): return bt.Bot.Risk

    config_file = build_live_trading_config(used_backtests, risk_plain)

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


@router.get("/portfolios/{portfolio_id}")
async def update_get(request: Request, portfolio_id: int):

    try:
        portfolio = portfolio_service.get_by_id(portfolio_id)
        portfolio_vm = PortfolioVM.model_validate(portfolio)

        return templates.TemplateResponse("/portfolios/update.html", {"request": request, "portfolio": portfolio_vm})
        
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/portfolios/{portfolio_id}")
async def update_post(
    request:Request,
    id:int = Form(...),
    name: str = Form(...),
    description: str = Form(...),
):
    try:
        _ = portfolio_service.update(id=id, name=name, description=description)
        return RedirectResponse(url="/portfolios", status_code=303)
        
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.post("/portfolios/run/{portfolio_id}")
async def run_portfolio(
    request: Request, 
    portfolio_id: int, 
    date_from:str=Form(...), 
    date_to:str=Form(...), 
    risk: str = Form(default=None)
):
    try:
        risk = float(risk) if risk else None

        date_from = Timestamp(date_from, tz="UTC")
        date_to = Timestamp(date_to, tz="UTC")

        portfolio = portfolio_service.get_by_id(portfolio_id)

        initial_cash = float(config_service.get_by_name('InitialCash').Value)

        strategies, tickers, timeframes, risks = [], [], [], []

        portfolio_backtests = portfolio_service.get_portfolio_backtests(portfolio_id=portfolio_id)
        for pf_bt in portfolio_backtests:
            strategies.append(pf_bt.BotPerformance.Bot.Strategy)
            tickers.append(pf_bt.BotPerformance.Bot.Ticker)
            timeframes.append(pf_bt.BotPerformance.Bot.Timeframe)

            risk = pf_bt.Weight if pf_bt.Weight else pf_bt.BotPerformance.Bot.Risk
            risks.append(risk)

        queue = asyncio.Queue()

        new_backtests = await backtest_service.run_backtests_and_save(
            initial_cash, 
            strategies, 
            tickers, 
            timeframes,
            date_from, 
            date_to, 
            'pa', 
            risks, 
            save_bt_plot='discard', 
            queue=queue,
            portfolio=True
        )

        portfolio_name = f'{portfolio.Name} {date_from.strftime("%Y%m%d")} - {date_to.strftime("%Y%m%d")}'
        description = f'Automated running for {portfolio.Name} from {date_from.strftime("%Y%m%d")} to {date_to.strftime("%Y%m%d")}'
        op_result = portfolio_service.create(name=portfolio_name, description=description)

        if op_result.ok:
            new_portfolio = op_result.item

        for bt in new_backtests:
            portfolio_service.add_performance(new_portfolio.Id, bt.Id, bt.Bot.Risk)

        return RedirectResponse(f"/portfolios/admin/{new_portfolio.Id}", status_code=303)
    
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})


@router.post("/portfolios/metrics/{portfolio_id}", response_class=HTMLResponse)
async def update_metrics(request: Request, portfolio_id:int):
    try:
        portfolio = portfolio_service.get_by_id(portfolio_id=portfolio_id)
        if not portfolio:
            return templates.TemplateResponse("/error.html", {"request": request})
        
        # Obtengo todos los backtest de un portfolio
        used_backtests = portfolio_service.get_backtests_from_portfolio(portfolio_id=portfolio_id)
        
        if not used_backtests:
            return templates.TemplateResponse(
            "/portfolios/portfolio_metrics.html", 
            {
                "request": request, 
                'metrics': None,
                'equity_plot': None
            }
        )
        
        used_backtests = [PerformanceMetricsVM.model_validate(backtest) for backtest in used_backtests]
        
        # obtengo las equity curves de todos los backtest en formato df
        trades_with_equity = portfolio_service.get_df_trades(portfolio_id)

        # Obtengo la curva de equity del portfolio
        portfolio_equity_curve = system_portfolio_metrics_service.get_full_equity_curve(trades_with_equity)
        
        system_portfolio_metrics_service.calculate_metrics_and_save(
            portfolio_id=portfolio_id, 
            equity_curve=portfolio_equity_curve, 
            all_trades=trades_with_equity
        )

        trades_with_equity[portfolio.Name] = portfolio_equity_curve
        equity_plot = system_portfolio_metrics_service.get_full_equity_curve_plot(trades_with_equity)
        plot_path = './app/templates/static/portfolio_plots'
        file_name = f'{portfolio_id}.html'
        with open(os.path.join(plot_path, file_name), 'w') as f:
            f.write(equity_plot)

        return RedirectResponse(f"/portfolios/admin/{portfolio_id}", status_code=303)

    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})


@router.post("/portfolios/admin/{portfolio_id}/report")
async def portfolio_report(request: Request, portfolio_id:int):
    try:
            portfolio = portfolio_service.get_by_id(portfolio_id=portfolio_id)
            if not portfolio:
                result = OperationResultVM(ok=False, message='The portfolio does not exists', item=None)
                return JSONResponse(result.model_dump_json())
            
            # Obtengo todos los backtest de un portfolio
            used_backtests = portfolio_service.get_backtests_from_portfolio(portfolio_id=portfolio_id)
            
            if not used_backtests:
                result = OperationResultVM(ok=False, message='The portfolio is empty', item=None)
                return JSONResponse(result.model_dump_json())

            
            used_backtests = [PerformanceMetricsVM.model_validate(backtest) for backtest in used_backtests]
            
            # obtengo las equity curves de todos los backtest en formato df
            trades_with_equity = portfolio_service.get_df_trades(portfolio_id)

            # Obtengo la curva de equity del portfolio
            portfolio_equity_curve = system_portfolio_metrics_service.get_full_equity_curve(trades_with_equity)
            print(portfolio_equity_curve)

            returns = portfolio_equity_curve.Equity.pct_change().dropna()

            plot_path = './app/templates/static/portfolio_reports'
            file_name = f'portfolio_{portfolio.Id}.html'
            full_path = os.path.join(plot_path, file_name)

            risk_free_rate = float(config_service.get_by_name('RiskFreeRate').Value)

            qs.reports.html(
                returns, 
                output=full_path, 
                title=portfolio.Name, 
                rf=risk_free_rate
            )

            return JSONResponse({"url": f"/static/portfolio_reports/{file_name}"})

    except Exception as e:
        result = OperationResultVM(ok=False, message='There was an error', item=None)
        return JSONResponse(result.model_dump_json())


@router.post("/portfolios/admin/merge")
async def merge_portfolios(
    request: Request,
    name: str = Form(...),  # Nombre del nuevo portfolio
    description: str = Form(default=""),  # Descripción opcional
    portfolio_ids: str = Form(...)  # IDs separados por comas
):
    try:
        # Convertir los IDs a lista de enteros
        ids_list = [int(id_str) for id_str in portfolio_ids.split(",") if id_str.strip()]
        
        if len(ids_list) < 2:
            raise HTTPException(status_code=400, detail="Select at least 2 portfolios to merge")
        
        op_result = portfolio_service.create(name=name, description=description)

        if op_result.ok:
            new_portfolio = op_result.item

        for portfolio_id in ids_list:

            pf_backtests = portfolio_service.get_backtests_from_portfolio(portfolio_id=portfolio_id)
            for bt in pf_backtests:
                portfolio_service.add_performance(new_portfolio.Id, bt.Id)
        
        return RedirectResponse(f"/portfolios/admin/{new_portfolio.Id}", status_code=303)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios/admin/{portfolio_id}/weights", response_class=HTMLResponse)
def get_weights(request:Request, portfolio_id:int):
    try:
        
        portfolio_backtests = portfolio_service.get_portfolio_backtests(portfolio_id=portfolio_id)

        portfolio_backtests = sorted(
            portfolio_backtests, 
            key=lambda pf_bt: pf_bt.BotPerformance.SharpeRatio, 
            reverse=True
        )

        portfolio_backtests_vm = []
        for pf_bt  in portfolio_backtests:
            portfolio_backtest_weights_vm = PortfolioBacktestVM(
                BacktestId=pf_bt.BotPerformance.Id,
                BotName=pf_bt.BotPerformance.Bot.Name, 
                BotWeight= 0 if not pf_bt.Weight else pf_bt.Weight
            )

            portfolio_backtests_vm.append(portfolio_backtest_weights_vm)

        update_portfolio_backtest_weights_vm = UpdatePortfolioBacktestVM(
            PortfolioId=portfolio_id, PortfolioBacktests=portfolio_backtests_vm
        )     

        return templates.TemplateResponse("/portfolios/modal_weights.html", {
            "request": request,
            'vm': update_portfolio_backtest_weights_vm
        })
    
    except Exception as e:
        logger.info(f'Hubo un error cargando el portfolio {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.post("/portfolios/admin/{portfolio_id}/weights")
def set_weights(portfolio_id: int, portfolio_backtests: List[PortfolioBacktestVM]):
    try:
        op_result = portfolio_service.set_bot_weights(
            portfolio_id=portfolio_id,
            bot_performance_ids=[pf_bt.BacktestId for pf_bt in portfolio_backtests],
            weights=[pf_bt.BotWeight for pf_bt in portfolio_backtests],
        )

        op_result = OperationResultVM.model_validate(op_result)

        return JSONResponse(content=op_result.model_dump())     

    except Exception as e:
        logger.info(f'Error {str(e)}')
        result = OperationResultVM(ok=False, message='There was an error..', item=None)
        return JSONResponse(content=result.model_dump())