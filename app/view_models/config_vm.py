from uuid import UUID
from pydantic import BaseModel, ConfigDict


class ConfigVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    Name: str
    Value:object
