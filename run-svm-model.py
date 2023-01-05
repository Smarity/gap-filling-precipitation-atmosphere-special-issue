#%%
import pandas as pd
import numpy as np
import pickle

from rmse import rmse

model = pickle.load(open("models/svm__ARJONA.sav", 'rb'))

data = pd.read_excel("data/inland-example-data.xlsx")

conf = [
    "DOY", "PRECIP_LINARES", 'DISTANCE_ARJONA-LINARES', "PRECIP_MANCHAREAL", 'DISTANCE_ARJONA-MANCHAREAL', 
    "PRECIP_MARMOLEJO", 'DISTANCE_ARJONA-MARMOLEJO', "PRECIP_SABIOTE", 'DISTANCE_ARJONA-SABIOTE', 
    "PRECIP_TORREBLASCOPEDRO", 'DISTANCE_ARJONA-TORREBLASCOPEDRO', "PRECIP_JAEN", 'DISTANCE_ARJONA-JAEN' ,
    "PRECIP_ARJONA"]

data_output = data[conf[-1]]
data_input = data.filter(items=conf[:-1])

scaler = pickle.load(open('scalers/scaler__ARJONA.pkl', 'rb'))
data_input_scaled = scaler.transform(data_input)

y_pred = np.array(model.predict(data_input_scaled))
y_pred = np.ravel(y_pred)

rmse = rmse(data_output, y_pred)
print(rmse)


# %%
