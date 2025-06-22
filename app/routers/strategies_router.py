import asyncio
import os
from typing import Annotated, AsyncGenerator
from uuid import uuid4
from fastapi import APIRouter, Body, Depends
from fastapi import Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pandas import Timestamp
from app.backbone.services.backtest_service import BacktestService
from app.backbone.services.config_service import ConfigService
from app.backbone.services.ticker_service import TickerService
from app.view_models.backtest_strategy import BacktestStrategyCreateVM
from app.view_models.category_vm import CategoryVM
from app.view_models.strategy_vm import StrategyVM
from app.backbone.services.strategy_service import StrategyService
from app.backbone.utils.logger import get_logger
from app.view_models.timeframe_vm import TimeframeVM

logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="./app/templates")
strategy_service = StrategyService()
ticker_service = TickerService()
backtest_service = BacktestService()
config_service = ConfigService()


# Ruta GET: muestra el formulario
@router.get("/strategies", response_class=HTMLResponse)
async def form_page(request: Request):
    try:
        logger.info("Obteniendo estrategias disponibles")

        strategies = strategy_service.get_all()
        
        strategies = [StrategyVM.model_validate(strategy) for strategy in strategies]
        strategies.sort(key=lambda x: x.Name, reverse=False)
        return templates.TemplateResponse("/strategies/index.html", {"request": request, "strategies": strategies})

    except Exception as e:
        logger.error(f'Hubo un error: {e}')
        return templates.TemplateResponse("/error.html", {"request": request})

# Ruta GET: muestra el formulario
@router.get("/strategies/new", response_class=HTMLResponse)
async def create_get(request: Request):
    try:
        return templates.TemplateResponse("/strategies/create.html", {"request": request})
    
    except Exception as e:
        return templates.TemplateResponse("/error.html", {"request": request})

# Ruta POST: Procesa datos del formulario
@router.post("/strategies")
async def create_post(
    request:Request,
    name: str = Form(...),
    description: str = Form(...),
    MetaTraderName: str = Form(...),
):
    try:
        _ = strategy_service.create(name=name, description=description, metatrader_name=MetaTraderName)
        return RedirectResponse(url="/strategies", status_code=303)

    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})
    
@router.get("/strategies/{strategy_id}")
async def update_get(request: Request, strategy_id: int):

    try:
        strategy = strategy_service.get_by_id(strategy_id)

        strategy_vm = StrategyVM.model_validate(strategy)

        return templates.TemplateResponse("/strategies/update.html", {"request": request, "strategy": strategy_vm})
    
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/strategies/{strategy_id}")
async def update_post(
    request:Request,
    id:int = Form(...),
    name: str = Form(...),
    description: str = Form(...),
    metatrader_name: str = Form(...),
):
    try:
        code_path =  './app/backbone/strategies'
        old_strategy = strategy_service.get_by_id(id=id)

        old_strategy_file_path = os.path.join(
            code_path, 
            old_strategy.Name.split('.')[0] + '.py'
        )

        if os.path.exists(old_strategy_file_path):
            new_strategy_code_file_path = os.path.join(
                code_path,
                name.split('.')[0] + '.py'
            )

            os.rename(old_strategy_file_path, new_strategy_code_file_path)

        _ = strategy_service.update(id=id, name=name, description=description, metatrader_name=metatrader_name)

        return RedirectResponse(url="/strategies", status_code=303)
        
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})

@router.post("/strategies/{strategy_id}/delete")
async def delete(request: Request, strategy_id: int):
    try:
        result = strategy_service.delete(strategy_id=strategy_id)
        if result.ok:
            return RedirectResponse(url="/strategies/", status_code=303)

        else:
            return templates.TemplateResponse("/error.html", {"request": request, "error": result.message})
    
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})

def normalize_code(code):
    normalized_code = code.replace('\r\n', '\n')
    return normalized_code

@router.get("/strategies/{strategy_id}/code")
async def get_strategy_code(request: Request, strategy_id: int):

    try:
        strategy = strategy_service.get_by_id(strategy_id)

        timeframes = ticker_service.get_all_timeframes()
        strategies = strategy_service.get_all()
        categories = ticker_service.get_all_categories()

        vm = BacktestStrategyCreateVM()

        vm.StrategyId = strategy.Id

        vm.InitialCash = config_service.get_by_name('InitialCash').Value
        vm.DateFrom = config_service.get_by_name('DateFromBacktest').Value
        vm.DateTo = config_service.get_by_name('DateToBacktest').Value
        vm.Risk = 1

        vm.Strategies = [StrategyVM.model_validate(strategy) for strategy in strategies]
        vm.Timeframes = [TimeframeVM.model_validate(timeframe) for timeframe in timeframes]
        vm.Categories = [CategoryVM.model_validate(category) for category in categories]

        file_name, class_name = strategy.Name.split('.')

        file_path = f'./app/backbone/strategies/{file_name}.py'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:  # Modo 'r' para solo lectura
                vm.Code = normalize_code(f.read())  # Leer todo el contenido del archivo

        return templates.TemplateResponse("/strategies/code.html", {"request": request, "model": vm})
    
    except Exception as e:
        logger.info(f"Hubo un error {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})

tasks = {}  # Guardará los generadores activos
async def event_generator(queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Lee la cola y envía los logs al frontend en tiempo real."""
    while True:
        message = await queue.get()
        if message == "DONE":
            break  # Finalizar el streaming cuando termine el proceso

        yield f"data: {message}\n\n"

@router.post("/strategies/{strategy_id}/code")
async def post_strategy_code(vm: Annotated[BacktestStrategyCreateVM, Form()]):
    
    task_id = str(uuid4())  # Generar un ID único para la tarea
    queue = asyncio.Queue()

    strategy = strategy_service.get_by_id(vm.StrategyId) # siempre va a existir pq se crea antes de escribir el codigo
    ticker = ticker_service.get_ticker_by_id(id=vm.TickerId)
    timeframe = ticker_service.get_timeframe_by_id(id=vm.TimeframeId)

    date_from = Timestamp(vm.DateFrom, tz="UTC")
    date_to = Timestamp(vm.DateTo, tz="UTC")

    file_name, class_name = strategy.Name.split('.')

    vm.Code = normalize_code(vm.Code)

    with open(f'./app/backbone/strategies/{file_name}.py', 'w+') as f:
        f.write(vm.Code)

    asyncio.create_task(backtest_service.run_backtest(
        initial_cash=vm.InitialCash,
        strategy=strategy,
        ticker=ticker,
        timeframe=timeframe,
        date_from=date_from,
        date_to=date_to,
        method='pa',
        risk=vm.Risk,
        save_bt_plot='temp',
        queue=queue,
    ))

    # Guardar la tarea en memoria
    tasks[task_id] = event_generator(queue)
    return {"task_id": task_id}


@router.get("/strategies/stream/{task_id}")
async def stream_results(task_id: str):
    if task_id not in tasks:
        return JSONResponse({"error": "Tarea no encontrada"}, status_code=404)
    return StreamingResponse(tasks[task_id], media_type="text/event-stream")


@router.post("/strategies/delete_plots/{strategy_id}")
async def delete_plots(strategy_id: int, request: Request):
    strategy = strategy_service.get_by_id(strategy_id) # siempre va a existir pq se crea antes de escribir el codigo

    try:
        body = await request.json()
        code = body.get("code", "")
        file_name, _ = strategy.Name.split('.')
        code = normalize_code(code)
        with open(f'./app/backbone/strategies/{file_name}.py', 'w+') as f:
            f.write(code)
    
    except Exception:
        pass


    plot_path = './app/templates/static/backtest_plots/temp'
    files_in_directory = os.listdir(plot_path)

    for file in files_in_directory:
        path_to_file = os.path.join(plot_path, file)
        os.remove(path_to_file)

    return {"status": "ok"}

    