import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
Df = pd.read_csv('./data/car-price-data.csv')
print(Df)
# data Cleaning
# removiing those col from the lis t
inputs = Df.drop(["Brand", "owner", "seller_type"], axis=1)
print(inputs)
# separating the selling price
target = Df.Sellig_price_in_ETB
print("Showing Target Selling Price \n",target)
# replacing str with numerical data  aka Conversion
Numerics = LabelEncoder()
inputs['Fuel_type_new'] = Numerics.fit_transform(inputs['Fuel_type'])
inputs['Transmissions_new'] = Numerics.fit_transform(inputs['Transmissions'])
print(inputs)
#droping string col
inputs_n=inputs.drop(["Fuel_type","Transmissions","Sellig_price_in_ETB"],axis=1)
print(inputs_n)
# linear reg
model = linear_model.LinearRegression()
# Training
print(model.fit(inputs_n, target))
predict_val = model.predict([[2020, 4000000 ,9000, 1, 0]])
print("Predicted Value in Birr : ",predict_val)   