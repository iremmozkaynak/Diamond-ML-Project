from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel, field_validator
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import logging

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Diamond Price Predictor", version="1.0.0")

# Templates
templates = Jinja2Templates(directory="templates")

# Model ve ön işleyicileri yükle
try:
    with open("30-diamond_model_complete.pkl", "rb") as f:
        saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    scaler = saved_data["scaler"]
    logger.info("Model başarıyla yüklendi.")
except FileNotFoundError:
    logger.error("Model dosyası bulunamadı: 30-diamond_model_complete.pkl")
    model = encoders = scaler = None
except KeyError as e:
    logger.error(f"Model dosyasında eksik anahtar: {e}")
    model = encoders = scaler = None
except Exception as e:
    logger.error(f"Model yüklenirken hata oluştu: {e}")
    model = encoders = scaler = None


class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

    @field_validator("carat", "depth", "table", "x", "y", "z")
    @classmethod
    def must_be_positive(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} pozitif bir sayı olmalıdır.")
        return v

    @field_validator("cut")
    @classmethod
    def valid_cut(cls, v):
        valid = {"Fair", "Good", "Very Good", "Premium", "Ideal"}
        if v not in valid:
            raise ValueError(f"Geçersiz cut değeri. Geçerli değerler: {valid}")
        return v

    @field_validator("color")
    @classmethod
    def valid_color(cls, v):
        valid = {"D", "E", "F", "G", "H", "I", "J"}
        if v not in valid:
            raise ValueError(f"Geçersiz color değeri. Geçerli değerler: {valid}")
        return v

    @field_validator("clarity")
    @classmethod
    def valid_clarity(cls, v):
        valid = {"I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"}
        if v not in valid:
            raise ValueError(f"Geçersiz clarity değeri. Geçerli değerler: {valid}")
        return v


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(features: DiamondFeatures):
    if model is None or encoders is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
        )

    try:
        input_data = pd.DataFrame([features.model_dump()])

        for col in ["cut", "color", "clarity"]:
            input_data[col] = encoders[col].transform(input_data[col])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        predicted_price = float(prediction[0])

        logger.info(f"Tahmin yapıldı: ${predicted_price:.2f}")
        return {"predicted_price": predicted_price, "status": "success"}

    except ValueError as e:
        logger.error(f"Değer hatası: {e}")
        raise HTTPException(status_code=400, detail=f"Geçersiz giriş verisi: {str(e)}")
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail="Tahmin sırasında bir hata oluştu.")
