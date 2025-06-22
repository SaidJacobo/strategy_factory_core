import logging

# Configurar el logging
logging.basicConfig(
    level=logging.INFO,  # Nivel mínimo a registrar (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato de salida
    handlers=[
        logging.FileHandler("app.log"),  # Guarda los logs en un archivo
        logging.StreamHandler()  # Muestra logs en la consola
    ]
)

# Crear un logger reutilizable
def get_logger(name):
    return logging.getLogger(name)
