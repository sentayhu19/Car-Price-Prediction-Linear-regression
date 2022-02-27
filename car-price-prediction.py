import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
Df = pd.read_csv('./data/car-price-data.csv')
# print(Df)
# data Cleaning
# removiing those col from the list
inputs = Df.drop(["Brand", "Present_Price", "owner", "seller_type"], axis=1)
# print(inputs)
# separating the selling price
target = Df.Sellig_price_in_ETB
print(target)
# replacing str with numerical data  aka Conversion
Numerics = LabelEncoder()
inputs['Fuel_type'] = Numerics.fit_transform(inputs['Fuel_type'])
inputs['Transmissions'] = Numerics.fit_transform(inputs['Transmissions'])
print(inputs)
# linear reg
model = linear_model.LinearRegression()
# Training
model.fit(inputs, target)
predict_val = model.predict([[2021, 600000, 27000, 0, 0]])
print(predict_val)
