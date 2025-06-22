from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class OperationResultVM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ok:bool
    message: Optional[str]
    item: Optional[object]