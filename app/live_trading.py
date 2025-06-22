import os
import sys
import logging
import subprocess
import yaml
import threading
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuraciones de rutas
CONFIG_PATH = './app/configs/live_trading_auto.yml'
WATCH_FOLDER = os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
LOG_FOLDER = "./app/logs"

if not os.path.exists(LOG_FOLDER):
    os.mkdir(LOG_FOLDER)

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_FOLDER, "live_trading.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Función para cargar configuración
def cargar_configuraciones():
    logger.info("Cargando configuraciones desde el YAML")
    with open(CONFIG_PATH, 'r') as file:
        return yaml.safe_load(file)

# Lanzar bot
def lanzar_bot(strategy_name, bot_name, ticker, timeframe, risk, opt_params, wfo_params, metatrader_name, tz, metatrader_folder_to_watch):
    stdout_log_path = f"{LOG_FOLDER}/{bot_name}_{ticker}_{timeframe}_stdout.log"
    stderr_log_path = f"{LOG_FOLDER}/{bot_name}_{ticker}_{timeframe}_stderr.log"

    with open(stdout_log_path, "w+") as stdout_log, open(stderr_log_path, "w+") as stderr_log:
        subprocess.Popen(
            [
                sys.executable,
                "./app/bot_runner.py",
                strategy_name,
                bot_name,
                ticker,
                timeframe,
                str(risk),
                yaml.dump(opt_params) if opt_params is not None else '{}',
                yaml.dump(wfo_params) if wfo_params is not None else '{}',
                metatrader_name,
                tz,
                metatrader_folder_to_watch
            ],
            stdout=stdout_log,
            stderr=stderr_log
        )

    logger.info(f"Bot {bot_name} lanzado con logs en {stdout_log_path} y {stderr_log_path}")

# Calcular hash de archivo
def hash_archivo(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"No se pudo calcular hash de {path}: {e}")
        return None

# Handler
class TickerFileHandler(FileSystemEventHandler):
    def __init__(self, configuraciones):
        self.configuraciones = configuraciones
        self.modified_files = {}  # filename -> last modified timestamp
        self.hashes_anteriores = {}  # filename -> hash
        self.lock = threading.Lock()
        self.debounce_interval = 0.5
        self.running = True
        threading.Thread(target=self.revisor_loop, daemon=True).start()
        logger.info("Watcher de archivos iniciado.")

    def on_modified(self, event):
        if event.is_directory:
            return
        filename = os.path.basename(event.src_path)
        if not filename.endswith(".csv"):
            return
        logger.info(f"Archivo modificado detectado: {filename}")
        with self.lock:
            self.modified_files[filename] = time.time()

    def revisor_loop(self):
        while self.running:
            time.sleep(0.1)
            now = time.time()
            with self.lock:
                archivos_a_procesar = [
                    f for f, t in self.modified_files.items()
                    if now - t > self.debounce_interval
                ]
                for f in archivos_a_procesar:
                    self.ejecutar_bots_para_archivo(f)
                    del self.modified_files[f]

    def ejecutar_bots_para_archivo(self, filename):
        filepath = os.path.join(WATCH_FOLDER, filename)
        nuevo_hash = hash_archivo(filepath)
        hash_anterior = self.hashes_anteriores.get(filename)

        if nuevo_hash is None:
            logger.warning(f"Archivo {filename} no se pudo leer. Se omite ejecución.")
            return

        if nuevo_hash == hash_anterior:
            logger.info(f"Ignorando {filename}: contenido no cambió (hash idéntico)")
            return

        self.hashes_anteriores[filename] = nuevo_hash
        logger.info(f"Ejecutando bots para archivo consolidado: {filename}")

        try:
            ticker, timeframe = filename.rsplit('_', 1)
            timeframe = timeframe.replace(".csv", "")
        except ValueError:
            logger.error(f"Nombre de archivo inválido: {filename}")
            return

        for strategy_name, configs in self.configuraciones.items():
            instruments_info = configs.get('instruments_info', {})
            if ticker not in instruments_info:
                continue

            info = instruments_info[ticker]
            if info['timeframe'] != timeframe:
                continue

            bot_name = strategy_name.split('.')[-1]
            logger.info(f"Lanzando bot {bot_name} para {ticker} en timeframe {timeframe}")
            lanzar_bot(
                strategy_name,
                bot_name,
                ticker,
                info['timeframe'],
                info['risk'],
                configs.get('opt_params', {}),
                configs.get('wfo_params', {}),
                configs['metatrader_name'],
                'Etc/UTC',
                WATCH_FOLDER
            )

# Main
if __name__ == "__main__":
    logger.info("Iniciando el servicio de monitoreo de archivos...")
    configuraciones = cargar_configuraciones()
    event_handler = TickerFileHandler(configuraciones)
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        observer.join()
    except KeyboardInterrupt:
        logger.info("Deteniendo el servicio de monitoreo de archivos...")
        event_handler.running = False
        observer.stop()
        observer.join()
    logger.info("Servicio finalizado.")
