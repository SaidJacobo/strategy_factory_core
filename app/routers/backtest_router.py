import asyncio
from datetime import date
import json
import os
from typing import Annotated, AsyncGenerator, Optional
from uuid import uuid4
from fastapi import APIRouter, Query
from fastapi import Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pandas import Timestamp
import yaml
from app.backbone.services.bot_service import BotService
from app.backbone.services.config_service import ConfigService
from app.backbone.services.strategy_service import StrategyService
from app.backbone.services.test_service import TestService
from app.backbone.utils.general_purpose import build_live_trading_config, save_ticker_timeframes
from app.view_models.backtest_create_vm import BacktestCreateVM
from app.view_models.performance_metrics_vm import PerformanceMetricsVM
from app.view_models.bot_performance_vm import BotPerformanceVM
from app.view_models.category_vm import CategoryVM
from app.view_models.config_vm import ConfigVM
from app.view_models.op_result_vm import OperationResultVM
from app.view_models.strategy_vm import StrategyVM
from app.view_models.ticker_vm import TickerVM
from app.view_models.timeframe_vm import TimeframeVM
from app.backbone.services.ticker_service import TickerService
from app.backbone.services.backtest_service import BacktestService
import plotly.graph_objects as go
from app.backbone.utils.logger import get_logger
from app.view_models.trade_vm import TradeVM
from dateutil import parser

logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="./app/templates")

ticker_service = TickerService()
strategy_service = StrategyService()
backtest_service = BacktestService()
test_service = TestService()
bot_service = BotService()
config_service = ConfigService()

@router.get("/backtest", response_class=HTMLResponse)
async def backtest_strategies(request: Request):
    try:
        strategies = strategy_service.get_used_strategies()
        strategies_vm = sorted(
            [StrategyVM.model_validate(strategy) for strategy in strategies], 
            key=lambda strategy: strategy.Name
        )
        
        return templates.TemplateResponse("/backtest/index.html", {"request": request, 'strategies': strategies_vm})
 
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
        
@router.get("/backtest/new", response_class=HTMLResponse)
async def create_get(request: Request):
    try:
        timeframes = ticker_service.get_all_timeframes()
        strategies = strategy_service.get_all()
        categories = ticker_service.get_all_categories()
    
        vm = BacktestCreateVM()

        vm.InitialCash = config_service.get_by_name('InitialCash').Value
        vm.Strategies = [StrategyVM.model_validate(strategy) for strategy in strategies]
        vm.Strategies.sort(key=lambda x: x.Name)
        vm.Categories = [CategoryVM.model_validate(category) for category in categories]

        vm.DateFrom = config_service.get_by_name('DateFromBacktest').Value
        vm.DateTo = config_service.get_by_name('DateToBacktest').Value
        
        timeframes_vm = [TimeframeVM.model_validate(timeframe) for timeframe in timeframes]
        for timeframe in timeframes_vm:
            setattr(vm, timeframe.Name, timeframe.Selected if timeframe.Selected != None else False)

        return templates.TemplateResponse("/backtest/create.html", {"request": request, "model": vm})
    
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

tasks = {}  # Guardará los generadores activos
async def event_generator(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Lee la cola y envía los logs al frontend en tiempo real."""
    while True:
        message = await queue.get()
        if message == "DONE":
            break  # Finalizar el streaming cuando termine el proceso

        yield f"data: {message}\n\n"
    
    tasks = {}

@router.post("/backtest")
async def create_post(backtest_vm: Annotated[BacktestCreateVM, Form()]):
    task_id = str(uuid4())  # Generar un ID único para la tarea
    queue = asyncio.Queue()

    # Obtener parámetros
    strategy_id = int(backtest_vm.StrategyId) if backtest_vm.StrategyId else None
    category_id = int(backtest_vm.CategoryId) if backtest_vm.CategoryId else None
    ticker_id = int(backtest_vm.TickerId) if backtest_vm.TickerId else None

    date_from = Timestamp(backtest_vm.DateFrom, tz="UTC")
    date_to = Timestamp(backtest_vm.DateTo, tz="UTC")

    timeframes = ticker_service.get_all_timeframes()
    selected_timeframes = [tf for tf in timeframes if getattr(backtest_vm, tf.Name)]

    strategy = None
    strategies = None

    if strategy_id:
        strategy = strategy_service.get_by_id(id=backtest_vm.StrategyId)
    else:
        strategies = strategy_service.get_all()

    if category_id is None:
        tickers = ticker_service.get_all_tickers()
    elif category_id and ticker_id is None:
        tickers = ticker_service.get_tickers_by_category(category_id=category_id)
    else:
        tickers = [ticker_service.get_ticker_by_id(id=ticker_id)]

    # Iniciar el backtest en segundo plano
    asyncio.create_task(backtest_service.run_backtests_and_save(
        backtest_vm.InitialCash, 
        strategy or strategies, 
        tickers, 
        selected_timeframes,
        date_from, 
        date_to, 
        'pa', 
        backtest_vm.Risk, 
        save_bt_plot= 'persist' if backtest_vm.SaveBtPyPlot is not None else 'discard', 
        queue=queue
    ))

    # Guardar la tarea en memoria
    tasks[task_id] = event_generator(queue)
    return {"task_id": task_id}

@router.get("/backtest/stream/{task_id}")
async def stream_results(task_id: str):
    if task_id not in tasks:
        return JSONResponse({"error": "Tarea no encontrada"}, status_code=404)
    return StreamingResponse(tasks[task_id], media_type="text/event-stream")

@router.get('/backtest/ticker/{ticker_id}')
async def get_backtest_by_ticker(request: Request, ticker_id: int, strategy_id: int = Query(...)):
    try:
        bot_performance = backtest_service.get_performances_by_strategy_ticker(ticker_id=ticker_id, strategy_id=strategy_id)
        bot_performance_vm = [PerformanceMetricsVM.model_validate(performance) for performance in bot_performance]

        return templates.TemplateResponse("/backtest/view_performances.html", {"request": request, "performances": bot_performance_vm})

    except:
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get('/backtest/bot/{bot_id}')
def get_bot_backtes(request: Request, bot_id: int, date_from: date = Query(...), date_to: date = Query(...)):
    try:
        bot_performance = backtest_service.get_performances_by_bot_dates(bot_id=bot_id, date_from=date_from, date_to=date_to)
            
        bot_performance_vm = BotPerformanceVM.model_validate(bot_performance)
        bot_performance_vm.TradeHistory = sorted(bot_performance_vm.TradeHistory, key=lambda trade: trade.ExitTime)

        # Equity plot
        dates = [trade.ExitTime for trade in bot_performance_vm.TradeHistory]
        equity = [trade.Equity for trade in bot_performance_vm.TradeHistory]
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates, 
                y=equity,
                name='equity original',
                mode="lines+markers", 
                line_shape="hv"
            )
        )
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Equity'
        )
        equity_plot = fig.to_json()
        
        str_date_from = str(bot_performance.DateFrom).replace('-','')
        str_date_to = str(bot_performance.DateTo).replace('-','')
        file_name=f'{bot_performance.Bot.Name}_{str_date_from}_{str_date_to}.html'
        
        luck_test_plot = None
        luck_plot_path = './app/templates/static/luck_test_plots'
        
        if os.path.exists(os.path.join(luck_plot_path, file_name)):
            with open(os.path.join(luck_plot_path, file_name), 'r') as f:
                luck_test_plot = json.load(f)  # Cargar el contenido JSON

        correlation_test_plot = None
        correlation_plot_path = './app/templates/static/correlation_plots'
        if os.path.exists(os.path.join(correlation_plot_path, file_name)):
            with open(os.path.join(correlation_plot_path, file_name), 'r') as f:
                correlation_test_plot = json.load(f)  # Cargar el contenido JSON
            bot_performance_vm.HasCorrelationTest = True

        t_test_plot = None
        t_plot_path = './app/templates/static/t_test_plots'
        if os.path.exists(os.path.join(t_plot_path, file_name)):
            with open(os.path.join(t_plot_path, file_name), 'r') as f:
                t_test_plot = json.load(f)  # Cargar el contenido JSON
            bot_performance_vm.HasTTest = True

        bt_py_path_plot = './app/templates/static/backtest_plots'
        if os.path.exists(os.path.join(bt_py_path_plot, file_name)):
            bot_performance_vm.HasBacktestingPyPlot = True

        report_path_plot = './app/templates/static/backtest_plots/reports'
        if os.path.exists(os.path.join(report_path_plot, file_name)):
            bot_performance_vm.HasReport = True


        return templates.TemplateResponse(
            "/backtest/view_bot_performance.html", 
            {
                "request": request, 
                "performance": bot_performance_vm, 
                'equity_plot': equity_plot,
                'luck_test_plot': luck_test_plot or {"data": [], "layout": {}},
                'correlation_test_plot': correlation_test_plot or {"data": [], "layout": {}},
                't_test_plot': t_test_plot or {"data": [], "layout": {}},
            }
        )
        
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.get('/backtest/{bot_performance_id}/montecarlo', response_class=HTMLResponse)
async def get_montecarlo_modal(request: Request, bot_performance_id: int):
    try:
        performance = {"Id": bot_performance_id}

        montecarlo_iterations = ConfigVM.model_validate(config_service.get_by_name(name='MontecarloIterations'))
        montecarlo_risk_of_ruin = ConfigVM.model_validate(config_service.get_by_name(name='MontecarloRiskOfRuin'))

        return templates.TemplateResponse("/backtest/montecarlo_form.html", {
            "request": request, 
            "performance": performance,
            "montecarlo_iterations": montecarlo_iterations,
            "montecarlo_risk_of_ruin": montecarlo_risk_of_ruin
        })
    
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post('/backtest/{bot_performance_id}/montecarlo')
def run_montecarlo_test(
    request: Request, 
    bot_performance_id:int,
    simulations: int = Form(...),
    threshold_ruin: float = Form(...),
):
    try:
        result = test_service.run_montecarlo_test(
            bot_performance_id=bot_performance_id, 
            n_simulations=simulations,
            threshold_ruin=threshold_ruin / 100
        )
        
        if result.ok:
            referer = request.headers.get('referer')  # Obtiene la URL de la página anterior
            return RedirectResponse(url=referer, status_code=303)

        else:
            return templates.TemplateResponse("/error.html", {"request": request, 'error': result.message})
        
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get('/backtest/{bot_performance_id}/luck_test', response_class=HTMLResponse)
async def get_luck_test_modal(request: Request, bot_performance_id: int):
    try:
        performance = {"Id": bot_performance_id}

        luck_test_percent_trades_to_remove = ConfigVM.model_validate(config_service.get_by_name(name='LuckTestPercentTradesToRemove'))

        return templates.TemplateResponse("/backtest/luck_test_form.html", {
            "request": request, 
            "performance": performance,
            "luck_test_percent_trades_to_remove": luck_test_percent_trades_to_remove
        })
    
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post('/backtest/{performance_id}/luck_test')
def run_luck_test(request: Request, performance_id:int, percent_trades_to_remove: int = Form(...)):
    try:
        result = test_service.run_luck_test(
            bot_performance_id=performance_id, 
            trades_percent_to_remove=percent_trades_to_remove 
        )
        
        if result.ok:
            referer = request.headers.get('referer')  # Obtiene la URL de la página anterior
            return RedirectResponse(url=referer, status_code=303)

        else:
            return templates.TemplateResponse("/error.html", {"request": request, 'error': result.message})

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get('/backtest/{bot_performance_id}/random_test', response_class=HTMLResponse)
async def get_random_test_modal(request: Request, bot_performance_id: int):
    try:
        performance = {"Id": bot_performance_id}

        random_test_iterations = ConfigVM.model_validate(config_service.get_by_name(name='RandomTestIterations'))

        return templates.TemplateResponse("/backtest/random_test_form.html", {
            "request": request, 
            "performance": performance,
            'random_test_iterations': random_test_iterations

        })

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post('/backtest/{performance_id}/random_test')
def run_random_test(request: Request, performance_id:int, iterations: int = Form(...)):
    try:
        result = test_service.run_random_test(performance_id, n_iterations=iterations)
        
        if result.ok:
            referer = request.headers.get('referer')  # Obtiene la URL de la página anterior
            return RedirectResponse(url=referer, status_code=303)

        else:
            return templates.TemplateResponse("/error.html", {"request": request, 'error': result.message})

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post('/backtest/{performance_id}/correlation_test')
def run_correlation_test(request: Request, performance_id:int):
    try:
        result = test_service.run_correlation_test(performance_id)
        
        if result.ok:
            referer = request.headers.get('referer')  # Obtiene la URL de la página anterior
            return RedirectResponse(url=referer, status_code=303)

        else:
            return templates.TemplateResponse("/error.html", {"request": request, 'error': result.message})
        
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.post('/backtest/{performance_id}/t_test')
def run_correlation_test(request: Request, performance_id:int):
    try:
        result = test_service.run_t_test(performance_id)
        
        if result.ok:
            referer = request.headers.get('referer')  # Obtiene la URL de la página anterior
            return RedirectResponse(url=referer, status_code=303)

        else:
            return templates.TemplateResponse("/error.html", {"request": request, 'error': result.message})
        
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post('/backtest/strategies/{strategy_id}/delete')
def delete_strategy_backtest(request: Request, strategy_id:int):
    try:
        result = backtest_service.delete_from_strategy(strategy_id)
        
        if result.ok:
            return RedirectResponse(url='/backtest', status_code=303)
        
        else:
            logger.info(f'Hubo un error {result.error}')
            return templates.TemplateResponse("/error.html", {"request": request})

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post('/backtest/{performance_id}/delete')
def delete_performance_id(request: Request, performance_id:int):
    try:
        result = backtest_service.delete(performance_id)
        
        if result.ok:
            referer = request.headers.get('referer')  # Obtiene la URL de la página anterior
            return RedirectResponse(url=referer, status_code=303)
        
        else:
            logger.info(f'Hubo un error {result.message}')
            return templates.TemplateResponse("/error.html", {"request": request})

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get('/backtest/strategies/{strategy_id}/get_robusts', response_class=HTMLResponse)
def get_robusts(request: Request, strategy_id:int):
    try:
        bot_performances = backtest_service.get_robusts_by_strategy_id(strategy_id=strategy_id)
        bot_performances_vm = [PerformanceMetricsVM.model_validate(performance) for performance in bot_performances]
        return templates.TemplateResponse("/backtest/modal_robust.html", {"request": request, "performances": bot_performances_vm})
        
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
     
@router.get("/backtest/strategies/{strategy_id}/tickers")
async def get_tickers_by_strategy(request: Request, strategy_id: int):
    try:
        result = ticker_service.get_tickers_by_strategy(strategy_id)
        if result.ok:
            tickers = [TickerVM.model_validate(ticker) for ticker in result.item]
            
            return templates.TemplateResponse("/backtest/modal_tickers.html", {"request": request, "tickers": tickers, "strategy_id":strategy_id})
        else:
            logger.info(f'Hubo un error {str(e)}')
            return templates.TemplateResponse("/error.html", {"request": request})
        
    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.post("/backtest/{performance_id}/favorites")
async def update_favorites(performance_id: int):
    try:
        op_result = backtest_service.update_favorite(performance_id)
        result = OperationResultVM.model_validate(op_result)
        return JSONResponse(content=result.model_dump())

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        result = OperationResultVM(ok=False, message='Hubo un error al actualizar los favoritos', item=None)
        return JSONResponse(content=result.model_dump())

@router.get('/backtest/{bot_performance_id}/trades', response_class=HTMLResponse)
async def get_trades_modal(request: Request, bot_performance_id: int):
    try:

        trades = backtest_service.get_trades(bot_performance_id=bot_performance_id)
        trades_vm = [TradeVM.model_validate(trade) for trade in trades]

        return templates.TemplateResponse("/backtest/trades_modal.html", {
            "request": request, 
            "trades": trades_vm,
        })

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})

@router.get('/backtest/search')
async def get_backtest_by_ticker(
    request: Request,
    return_: Optional[str] = Query(None),
    drawdown: Optional[str] = Query(None),
    stability_ratio: Optional[str] = Query(None),
    sharpe_ratio: Optional[str] = Query(None),
    trades: Optional[str] = Query(None),
    rreturn_dd: Optional[str] = Query(None),
    custom_metric: Optional[str] = Query(None),
    winrate: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None),
):
    try:
        def parse(value): return float(value) if value not in (None, "") else None

        filters = {
            "return_": parse(return_),
            "drawdown": parse(drawdown),
            "stability_ratio": parse(stability_ratio),
            "sharpe_ratio": parse(sharpe_ratio),
            "trades": parse(trades),
            "rreturn_dd": parse(rreturn_dd),
            "custom_metric": parse(custom_metric),
            "winrate": parse(winrate),
            "strategy": strategy.strip() if strategy else None,
            "ticker": ticker.strip() if ticker else None,
        }

        if all(value is None for value in filters.values()):
            return templates.TemplateResponse("/backtest/search.html", {
                "request": request,
                "performances": []
            })

        bot_performances = backtest_service.get_by_filter(**filters)
        bot_performance_vm = [PerformanceMetricsVM.model_validate(p) for p in bot_performances]

        return templates.TemplateResponse("/backtest/search.html", {
            "request": request,
            "performances": bot_performance_vm
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post('/backtest/{performance_id}/deploy')
async def deploy_portfolio(request: Request, performance_id: int):

    try:
        backtest = backtest_service.get_bot_performance_by_id(bot_performance_id=performance_id)

        def risk_plain(bt): return bt.Bot.Risk

        config_file = build_live_trading_config(backtest, risk_plain)

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

        result = OperationResultVM(ok=True, message=None, item=None)
        return JSONResponse(content=result.model_dump())

    except Exception as e:
        result = OperationResultVM(ok=False, message=f'There was an error: {e}', item=None)
        return JSONResponse(content=result.model_dump())

 