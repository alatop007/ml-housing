import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory=".")

model = pickle.load(open("XGBmodel.pkl", "rb"))

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict-json/", response_class=JSONResponse)
async def predict_json(input_data: InputData):
    data = input_data.dict()
    input_values = list(data.values())
    new_data = np.array(input_values).reshape(1, -1)
    output = model.predict(new_data)
    return {"prediction": float(output[0])}

@app.post("/predict-form/", response_class=HTMLResponse)
async def predict_form(request: Request):
    form_data = await request.form()
    try:
        input_values = [float(form_data[field]) for field in form_data]
    except ValueError:
        return JSONResponse(content={"error": "Invalid input data"}, status_code=400)
    if len(input_values) != 8:
        return JSONResponse(content={"error": "Invalid number of input values"}, status_code=400)
    final_input = np.array(input_values).reshape(1, -1)
    output = model.predict(final_input)
    print(output)
    return templates.TemplateResponse("home.html", {"request": request, "prediction_text": f'The house price is {output[0]}'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)