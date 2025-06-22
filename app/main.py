from datetime import datetime, timedelta
import os
import subprocess
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    
import webbrowser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.routers import admin_router, backtest_router, categories_router, strategies_router
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.backbone.services.config_service import ConfigService
from app.backbone.services.ticker_service import TickerService

app = FastAPI()

app.include_router(strategies_router.router)
app.include_router(categories_router.router)
app.include_router(backtest_router.router)
app.include_router(admin_router.router)

templates = Jinja2Templates(directory="./app/templates")

app.mount("/static", StaticFiles(directory="./app/templates/static"), name="static")


backtests_plot_path = './app/templates/static/backtest_plots'
if not os.path.exists(backtests_plot_path):
    os.mkdir(backtests_plot_path)

reports_path = './app/templates/static/backtest_plots/reports'
if not os.path.exists(reports_path):
    os.mkdir(reports_path)

app.mount("/backtest_plots", StaticFiles(directory=backtests_plot_path), name="backtest_plots")
app.mount("/backtest_plots/reports", StaticFiles(directory=reports_path), name="reports_plots")

config_service = ConfigService()
ticker_service = TickerService()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    config_data = {
        "InitialCash": 100_000,
        "DateFromBacktest": (datetime.now() - timedelta(days=365 * 7)).strftime("%Y-%m-%d"),
        "DateToBacktest": (datetime.now() - timedelta(days=30 * 5)).strftime("%Y-%m-%d"),
        "TelegramBotToken": '',
        "TelegramBotChatId": '',
        "PositiveHitThreshold": 10,
        "NegativeHitThreshold": 10,
        "MontecarloIterations": 100,
        "MontecarloRiskOfRuin": 90,
        "RandomTestIterations": 100,
        "LuckTestPercentTradesToRemove": 5,
        "RiskFreeRate": 0.02,
    }

    # Guardar la configuración de forma iterativa
    for name, value in config_data.items():
        config = config_service.get_by_name(name=name)

        if not config:
            config_service.add_or_update(name=name, value=value)
    
    ticker_service.create_update_timeframes()

    return templates.TemplateResponse("home.html", {"request": request})


if __name__ == "__main__":

    subprocess.run(["alembic", "upgrade", "head"])
    subprocess.Popen([sys.executable, "./app/live_trading.py"], )
    webbrowser.open("http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
    