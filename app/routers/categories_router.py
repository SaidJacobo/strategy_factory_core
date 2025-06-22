from typing import List, Optional
from fastapi import APIRouter
from fastapi import Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.view_models.category_vm import CategoryVM
from app.view_models.op_result_vm import OperationResultVM
from app.view_models.ticker_vm import TickerVM
from app.backbone.services.ticker_service import TickerService
from app.backbone.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="./app/templates")
ticker_service = TickerService()

@router.get("/categories/", response_class=HTMLResponse)
async def categories(request: Request):
    try:
        categories = ticker_service.get_all_categories()
        categories_vm = [CategoryVM.model_validate(category) for category in categories]
        
        return templates.TemplateResponse("/tickers/categories.html", {"request": request, "categories": categories_vm})

    except Exception as e:
        logger.info(f'Hubo un error {str(e)}')
        return templates.TemplateResponse("/error.html", {"request": request})
        
        
@router.get('/categories/{category_id}/tickers')
def category_tickers(request: Request, category_id: int):
    result = ticker_service.get_tickers_by_category(category_id=category_id)

    try:
        tickers = [TickerVM.model_validate(ticker) for ticker in result]
        
        # Verifica el encabezado Accept
        if "application/json" in request.headers.get("Accept", ""):
            # Retorna JSON si se solicita
            return tickers  # FastAPI automáticamente serializa listas de Pydantic a JSON
        else:
            # Retorna HTML como antes
            return templates.TemplateResponse("/tickers/tickers.html", {"request": request, "tickers": tickers})
    
    except Exception as e:
        error_response = {"message": "Error", "data": result.message}
        if "application/json" in request.headers.get("Accept", ""):
            return error_response  # Respuesta JSON en caso de error
        else:
            return templates.TemplateResponse("/error.html", {"request": request, "error": error_response})


@router.post("/categories/update_tickers")
async def update_tickers():
    result = ticker_service.create()
    return RedirectResponse(url="/categories/", status_code=303)


@router.post("/categories/update_commissions")
async def update_commissions(categories: List[CategoryVM]):
    try:
        
        cat_ids = [cat.Id for cat in categories]
        commissions = [cat.Commission for cat in categories]

        op_result = ticker_service.update_categories_commissions(cat_ids, commissions)
        op_result = OperationResultVM.model_validate(op_result)

        return JSONResponse(content=op_result.model_dump())
    
    except Exception as e:
        logger.info(f'Error {str(e)}')
        result = OperationResultVM(ok=False, message='There was an error..', item=None)
        return JSONResponse(content=result.model_dump())

    
    
