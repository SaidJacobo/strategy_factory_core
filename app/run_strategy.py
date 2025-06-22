import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import yaml
from app.backbone.utils.general_purpose import load_function
import numpy as np
import logging
from pytz import timezone

np.seterr(divide='ignore')

WATCH_FOLDER = os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\Common\Files")

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,  # Nivel de logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Formato del mensaje
    handlers=[
        logging.FileHandler("run_strategy.log"),  # Archivo donde se guardan los logs
        logging.StreamHandler()  # Mostrar también en consola
    ]
)

if __name__ == '__main__':
    logger = logging.getLogger("run_strategy")

    tz = timezone('Etc/GMT-2')
    root = './backbone/data'
    
    with open('./app/configs/live_trading.yml', 'r') as file:
        strategies = yaml.safe_load(file)

    strategy_path = 'b_percent_strategy.BPercent'
    bot_path = 'app.bot_runner.BotRunner'
    selected_ticker = 'NZDUSD'
    
    configs = strategies[strategy_path]

    instruments_info = configs['instruments_info']
    wfo_params = configs['wfo_params']
    opt_params = configs['opt_params']
    
    for ticker, info in instruments_info.items():
        
        if ticker != selected_ticker:
            continue

        timeframe = info['timeframe']
        risk = info['risk']
        
        strategy = load_function('app.backbone.strategies.' + strategy_path)

        name = configs['metatrader_name']
        bot = load_function(bot_path)(
            name, 
            ticker, 
            timeframe, 
            opt_params, 
            wfo_params, 
            strategy, 
            risk, 
            tz,
            data_path=WATCH_FOLDER
        )
        
        bot.run()
