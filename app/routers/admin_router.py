from datetime import datetime, timedelta
from typing import Annotated, List, Optional
from fastapi import APIRouter
from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.backbone.services.config_service import ConfigService
from app.view_models.category_vm import CategoryVM
from app.view_models.config_update_vm import ConfigUpdateVM
from app.view_models.ticker_vm import TickerVM
from app.backbone.services.ticker_service import TickerService
from app.backbone.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="./app/templates")
ticker_service = TickerService()
config_service = ConfigService()

@router.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    try:
        initial_cash = config_service.get_by_name(name='InitialCash').Value
        date_from_backtest = config_service.get_by_name(name='DateFromBacktest').Value
        date_to_backtest = config_service.get_by_name(name='DateToBacktest').Value

        telegram_bot_token = config_service.get_by_name(name='TelegramBotToken').Value
        telegram_bot_chat_id = config_service.get_by_name(name='TelegramBotChatId').Value
        positive_hit_threshold = config_service.get_by_name(name='PositiveHitThreshold').Value
        negative_hit_threshold = config_service.get_by_name(name='NegativeHitThreshold').Value
        montecarlo_iterations = config_service.get_by_name(name="MontecarloIterations").Value
        montecarlo_risk_of_ruin = config_service.get_by_name(name="MontecarloRiskOfRuin").Value
        random_test_iterations = config_service.get_by_name(name="RandomTestIterations").Value
        luck_test_percent_trades_to_remove = config_service.get_by_name(name="LuckTestPercentTradesToRemove").Value

        risk_free_rate = config_service.get_by_name(name="RiskFreeRate").Value

        config_vm = ConfigUpdateVM(
            InitialCash = initial_cash,
            DateFromBacktest=date_from_backtest,
            DateToBacktest=date_to_backtest,
            TelegramBotToken = telegram_bot_token,
            TelegramBotChatId = telegram_bot_chat_id,
            PositiveHitThreshold = float(positive_hit_threshold),
            NegativeHitThreshold = float(negative_hit_threshold),
            MontecarloIterations = int(montecarlo_iterations),
            MontecarloRiskOfRuin = float(montecarlo_risk_of_ruin),
            RandomTestIterations = int(random_test_iterations),
            LuckTestPercentTradesToRemove = float(luck_test_percent_trades_to_remove),
            RiskFreeRate = float(risk_free_rate)
        )

        timeframes = ticker_service.get_all_timeframes()

        for timeframe in timeframes:
            setattr(config_vm, timeframe.Name, timeframe.Selected)

        return templates.TemplateResponse("/admin/update.html", {"request": request, 'configs': config_vm})

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
    

@router.post("/admin")
async def update_post(
    request: Request,
    config_vm: Annotated[ConfigUpdateVM, Form()]
):
    try:
        timeframes = ticker_service.get_all_timeframes()

        # Guardar la configuración de forma iterativa
        for name, value in config_vm.model_dump().items():
            timeframe = next((tf for tf in timeframes if tf.Name == name), None)

            if not timeframe:
                config_service.add_or_update(name=name, value=value)
            else:
                ticker_service.update_timeframe(timeframe, selected=value)

        return RedirectResponse(url="/admin", status_code=303)

    except Exception as e:
        logger.info(f"Hubo un error: {str(e)}")
        return templates.TemplateResponse("/error.html", {"request": request})
