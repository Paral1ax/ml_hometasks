# This is a sample Python script.
import pip
pip.main(['install', 'fastapi', 'pydantic'])
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


app = FastAPI()


class Item(BaseModel):
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


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return ...


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return ...


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass