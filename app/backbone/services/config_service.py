from typing import List
from app.backbone.database.db_service import DbService
from app.backbone.entities.config import Config
from app.backbone.services.operation_result import OperationResult

class ConfigService:
    def __init__(self):
        self.db_service = DbService()

    def get_all(self) -> List[Config]:
        with self.db_service.get_database() as db:
            configs = self.db_service.get_all(db, Config)

            return configs

    def get_by_name(self, name):
        with self.db_service.get_database() as db:
            config = self.db_service.get_by_filter(db, Config, Name=name)
            return config

    def add_or_update(self, name:str, value:str):
        with self.db_service.get_database() as db:
            config = self.db_service.get_by_filter(db, Config, Name=name)  

            # Si existe se modifica
            if config:
                new_config = Config(Id=config.Id, Name=name, Value=value)
                strategy = self.db_service.update(db, Config, new_config)
                result = OperationResult(ok=True, message=None, item=strategy)
                return result
            
            # Sino se agrega
            new_config = Config(Name=name, Value=value)
            strategy = self.db_service.create(db, new_config)
            result = OperationResult(ok=True, message=None, item=strategy)

            return result

    def load_default_values():
        pass