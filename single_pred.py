import logging
from pickle import load
import pandas as pd
from fastapi.exceptions import ResponseValidationError
from pydantic import ValidationError
from model import update_features, convert_poly_features, log


def preprocessing_df(df):
    try:
        transformer = load(open('transformer.pkl', 'rb'))
        df_scaler = load(open('scale.pkl', 'rb'))
        df[['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']] = update_features(df)
        df = df.dropna()
        df[['seats', 'engine']] = df[['seats', 'engine']].astype(int)
        if 'selling_price' in df.columns:
            df = df.drop(['name', 'selling_price'], axis=1)
        else:
            df.drop('name', axis=1)
        df_transformed = pd.DataFrame(transformer.transform(df),
                                      columns=transformer.get_feature_names_out())
        df_scaled = pd.DataFrame(df_scaler.transform(df_transformed),
                                 index=df_transformed.index, columns=df_transformed.columns)
        df_poly = convert_poly_features(df_scaled)
        return df_poly
    except Exception as exc:
        log.error(exc)


def item_predict(item):
    log.info("Single item prediction started")
    try:
        df = pd.DataFrame([vars(f) for f in [item]])
        df = preprocessing_df(df)
        load_model = load(open('model.pkl', 'rb'))
        log.info("Single item prediction successfully completed")
        return load_model.predict(df)
    except ValidationError | Exception as e:
        log.error(e)


def items_predict(items):
    logging.info("CSV prediction started")
    try:
        df = pd.DataFrame.from_records([item.__dict__ for item in items])
        df = preprocessing_df(df)
        load_model = load(open('model.pkl', 'rb'))
        log.info("CSV prediction successfully completed")
        pred = load_model.predict(df)
        return pred
    except Exception as e:
        log.error(e)


def csv_precit(file):
    logging.info("CSV prediction started")
    try:
        df_base = pd.read_csv(file)
        df_base = df_base.dropna()
        df = df_base.copy()
        df = preprocessing_df(df)
        load_model = load(open('model.pkl', 'rb'))
        log.info("CSV prediction successfully completed")
        pred = load_model.predict(df)
        df_base['selling_price'] = pred
        return df_base
    except Exception as e:
        log.error(e)
