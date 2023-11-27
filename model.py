import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


def download_and_drop_dupl():
    df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
    df_train = df_train.drop_duplicates(subset=df_train.drop(['selling_price'], axis=1), keep='first')
    df_train = df_train.reset_index().drop('index', axis=1)
    y_train = df_train['selling_price']
    return df_train, y_train


def update_features(df):
    # Тут тяжело. Заменяем все буквенные и пробельные символы на пустую строку
    df_new = df[['mileage', 'engine', 'max_power']].replace(r'[^\d*.\d*@-]', '', regex=True)
    # Преобразуем в числа
    df_exc_tor = df_new.apply(pd.to_numeric)
    # Заменяем все символы кроме чисел вкл . и , на /, и также убираем последнее вхождение /
    df_exc_tor['torq'] = df['torque'].replace(r'[^\d*.,]', '/', regex=True).replace(r'/+', '/', regex=True).replace(
        r'/$', '', regex=True)
    # Сплитим по / и берем 0 значение - это torque
    df_exc_tor['torque'] = df_exc_tor['torq'].apply(lambda t: str(t).split('/')[0])
    # Заменяем запятые на точки для успешного преобразования в float
    df_exc_tor['max_torque_rpm'] = df_exc_tor['torq'].replace(r',', '.', regex=True)
    # Сплитим по / и берем последнее значение - это max_torque_rpm
    df_exc_tor['max_torque_rpm'] = df_exc_tor['max_torque_rpm'].apply(lambda t: str(t).split('/').pop())
    # Убираем промежуточное значение из датафрейма
    df_exc_tor = df_exc_tor.drop('torq', axis=1)
    # Заполняем пропуски и битые значения на медиану
    df_exc_tor = df_exc_tor.fillna(df_exc_tor.median()).replace(r'nan', df_exc_tor.median(), regex=True)
    # Переводим оставшиеся значения в float
    df_exc_tor = df_exc_tor.apply(pd.to_numeric)

    return df_exc_tor


def preprocessing_df(df_train, y_train):
    df_train[['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']] = update_features(df_train)

    df_train['seats'] = df_train['seats'].fillna(df_train['seats'].median()).replace(r'nan', df_train['seats'].median(),
                                                                                     regex=True)
    df_train[['seats', 'engine']] = df_train[['seats', 'engine']].astype(int)

    X_train_cat = df_train.drop(['name', 'selling_price'], axis=1)
    x_train_dummies = pd.get_dummies(X_train_cat, columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats'])

    # x_test_dummies.insert(17,'seats_4', 0)
    # x_test_dummies[['seats_10', 'seats_14']] = 0

    df_scaler = StandardScaler()

    x_train_dummies = pd.DataFrame(df_scaler.fit_transform(x_train_dummies),
                                   index=x_train_dummies.index, columns=x_train_dummies.columns)

    df_concat = pd.concat([x_train_dummies, y_train], axis=1)

    return df_concat


# x^2
squared_features = ['year', 'engine', 'max_power', 'torque']
# 1/x
hyperbole_features = ['km_driven']


def convert_poly_features(df):
    poly_squared = PolynomialFeatures(2)
    squared = poly_squared.fit_transform(df[squared_features])
    df_squared_f = poly_squared.get_feature_names_out()
    df[df_squared_f] = squared
    df['km_driven^0.5'] = df['km_driven'].apply(lambda x: x ** -1)
    return df


def fit_model(df_concat):
    df_train_poly = convert_poly_features(df_concat.drop('selling_price', axis=1))

    ridge_grid_search = GridSearchCV(Ridge(), param_grid={'alpha': (np.logspace(-8, 8, 100))}, cv=5, scoring='r2',
                                     error_score='raise')

    grid_train = ridge_grid_search.fit(df_train_poly, df_concat['selling_price'])

    return grid_train


def full_transform_fitting_process():
    x_train, y_train = download_and_drop_dupl()
    x_train = preprocessing_df(x_train, y_train)
    fitted_model = fit_model(x_train)
    return fitted_model
