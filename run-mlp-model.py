#%%
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from rmse import rmse

model = tf.keras.models.load_model("models/mlp__ARJONA.sav")

data = pd.read_excel("data/inland-example-data.xlsx")

conf = [
    "DOY", "PRECIP_LINARES", 'DISTANCE_ARJONA-LINARES', "PRECIP_MANCHAREAL", 'DISTANCE_ARJONA-MANCHAREAL', 
    "PRECIP_MARMOLEJO", 'DISTANCE_ARJONA-MARMOLEJO', "PRECIP_SABIOTE", 'DISTANCE_ARJONA-SABIOTE', 
    "PRECIP_TORREBLASCOPEDRO", 'DISTANCE_ARJONA-TORREBLASCOPEDRO', "PRECIP_JAEN", 'DISTANCE_ARJONA-JAEN' ,
    "PRECIP_ARJONA"]

'''
conf for coastal locations

conf = [
    "DOY","PRECIP_ANTEQUERA", "DISTANCE_MALAGA-ANTEQUERA", "PRECIP_ARCHIDONA", "DISTANCE_MALAGA-ARCHIDONA", 
    "PRECIP_CARTAMA", "DISTANCE_MALAGA-CARTAMA", "PRECIP_CHURRIANA", "DISTANCE_MALAGA-CHURRIANA", 
    "PRECIP_PIZARRA", "DISTANCE_MALAGA-PIZARRA", "PRECIP_VELEZ", "DISTANCE_MALAGA-VELEZ", "PRECIP_MALAGA"]
'''

data_output = data[conf[-1]]
data_input = data.filter(items=conf[:-1])

scaler = pickle.load(open('scalers/scaler__ARJONA.pkl', 'rb'))
data_input_scaled = scaler.transform(data_input)

y_pred = np.array(model.predict(data_input_scaled))
y_pred = np.ravel(y_pred)

rmse = rmse(data_output, y_pred)
print(rmse)


# %%
