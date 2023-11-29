# ml_hometasks
ДЗ №1

Файл с ноутбуком называется *HW1_Regression_with_inference (1)*

Что было сделано:
1. Сначала выборка была преобразована с помощью OneHotEncoder и в ноутбуке и в файле .py (Данные преобразования находятся в файле *transformer.pkl*)
2. Затем я заскейлил фичи с помощью StandardScale и в ноутбуке и в файле .py (Данные скейлинга находятся в файле *scale.pkl*) 
3. Далее я попробовал избавиться от выбросов, но в конце модель плохо предсказывала,\
потому что выборка маленькая и не хочется из нее что то еще выкидывать
4. Затем с помощью PolynomialFeatures я добавил квадратичных фичей для ['year','engine','max_power', 'torque']\
и посмотрев на график, решил что можно ['km_driven'] преобразовать как 1/х
5. В конце я обучил Ridge модель с помощью GridSearchCV и сделал предикт (Данные весов находятся в файле *model.pkl*) 
6. Оказалось что 1/4 > предсказаний находится в пределах 10% от оригинала.

**ОПИСАНИЕ СЕРВИСА**\
В файле *main.py* находится сервис, с предобученной моделью
Пришлось переработать классы, чтобы они соответствовали описанию. Потому что для CSV (списка объектов)\
нужно вручную добавлять столбец с selling_price, а не принимать его с данными.

В файле *model.py* находится код для предобработки и обучения модели\
**Модель обучается при старте сервиса.**\
После этого все приходящие объекты обучаются на весах, которые приходят из файла *model.pkl*

**Работа с получаемыми объектами**\
Работа с объектами находится в файле *single_pred.py*\
Сначала объекты проходят трансформацию, через ранее скаченные файлы весов\
И обучаются на ранее найденных и скаченных весах модели.

**Логи из работы сервиса находятся в файле *model_hw1.log***

**Демонстрация работы сервиса**\
**1. метод */predict_items*:**\
Входные данные:\
[{
    "name":"aaa",\
    "year":2005,\
    "km_driven":210000,\
    "fuel":"Petrol",\
    "seller_type":"Individual",\
    "transmission":"Manual",\
    "owner":"First Owner",\
    "mileage":"18.5 kmpl",\
    "engine":"1197 CC",\
    "max_power":"82.85 bhp",\
    "torque":"90Nm@ 3500rpm",\
    "seats":5.0\
},\
{\
    "name":"BBB",\
    "year":2020,\
    "km_driven":210000,\
    "fuel":"Diesel",\
    "seller_type":"Individual",\
    "transmission":"Manual",\
    "owner":"First Owner",\
    "mileage":"22.5 kmpl",\
    "engine":"1000 CC",\
    "max_power":"82.85 bhp",\
    "torque":"90Nm@ 3500rpm",\
    "seats": 5.0\
},\
{\
    "name":"CCC",\
    "year":2015,\
    "km_driven":15000,\
    "fuel":"Diesel",\
    "seller_type":"Dealer",\
    "transmission":"Automatic",\
    "owner":"First Owner",\
    "mileage":"22.5 kmpl",\
    "engine":"1000 CC",\
    "max_power":"82.85 bhp",\
    "torque":"90Nm@ 3500rpm",\
    "seats": 5.0\
}]\
Выходные данные:\
[\
    24214.198621585616,\
    616563.0679812334,\
    639870.3205116787\
]\
**2. метод */predict_item*:** \
Входные данные: \
{\
    "name":"aaa",\ 
    "year":2010,\
    "selling_price": 0,\
    "km_driven":10000,\
    "fuel":"Petrol",\
    "seller_type":"Individual",\ 
    "transmission":"Manual",\
    "owner":"First Owner",\
    "mileage":"18.5 kmpl",\
    "engine":"1197 CC",\
    "max_power":"82.85 bhp",\ 
    "torque":"90Nm@ 3500rpm",\
    "seats":5.0\
}\
Выходные данные:\
193593.49411537085

**UPD**

**Добавил метод для обработки CSV файлов**

**Скрины**\
![image](https://github.com/Paral1ax/ml_hometasks/assets/71229854/f53b7cc3-f47a-4433-9a15-32ca28514459)

![image](https://github.com/Paral1ax/ml_hometasks/assets/71229854/03f238cc-1bd2-419b-9ce4-520add4699a2)

![image](https://github.com/Paral1ax/ml_hometasks/assets/71229854/19e98c57-3720-45f5-abb4-bf9fffce38b4)

![image](https://github.com/Paral1ax/ml_hometasks/assets/71229854/d50ffc67-1423-4275-9822-4cdf123ad2fb)

![image](https://github.com/Paral1ax/ml_hometasks/assets/71229854/b38d5cc0-4138-4a3d-b69d-d2914d4b97fc)

![image](https://github.com/Paral1ax/ml_hometasks/assets/71229854/6ca8df46-ec9e-43bb-94f3-91a42af8c87a)

