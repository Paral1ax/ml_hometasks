# This is a sample Python script.
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from model import full_transform_fitting_process
from single_pred import item_predict, items_predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    full_transform_fitting_process()
    yield


app = FastAPI(lifespan=lifespan)


class Single_item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Csv_items(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Csv_items]


@app.post("/predict_item")
def predict_item(item: Single_item) -> float:
    return item_predict(item)


@app.post("/predict_items")
def predict_items(items: List[Csv_items]) -> List[float]:
    return items_predict(items)
