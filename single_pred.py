import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from main import Item
from model import update_features

df_scaler = StandardScaler()


def preprocessing_df(item, fit_df):
    df = pd.DataFrame([vars(f) for f in [item]], columns=[fit_df.columns])
    df = df.fillna(0)
    features = ['seats_' + str(int(item.seats)), 'fuel_' + item.fuel, 'seller_type_' + item.seller_type,
                'transmission_' + item.transmission, 'owner_' + item.owner]
    df[features] = 1

def transform_single_obj(item: Item):
    df_object = pd.DataFrame(item)
    df_upd = update_features(df_object)
