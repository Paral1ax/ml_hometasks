import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from main import Item
from model import update_features


def preprocessing_df(df, column_names):
    df[['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']] = update_features(df)

    df[['seats', 'engine']] = df[['seats', 'engine']].astype(int)

    x_cat = df.drop(['name'], axis=1)
    x_dummies = pd.get_dummies(x_cat, columns=['fuel', 'seller_type', 'transmission', 'owner'],
                               drop_first=True)


    df_scaler = StandardScaler()

    x_dummies = pd.DataFrame(df_scaler.fit_transform(x_dummies),
                             index=x_dummies.index, columns=x_dummies.columns)

    return x_dummies


def transform_single_obj(item: Item):
    df_object = pd.DataFrame(item)
    df_upd = update_features(df_object)
