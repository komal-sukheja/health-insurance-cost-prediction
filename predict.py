import torch
import pandas as pd
import pickle
from model import InsuranceRegressionModel

# load encoders & scaler
labelencoder = pickle.load(open("encoders/labelencoders.pkl", "rb"))
scaler = pickle.load(open("encoders/scaler.pkl", "rb"))

# infer input dim from scaler
input_dim = scaler.mean_.shape[0]

# load model
model = InsuranceRegressionModel(input_dim)
model.load_state_dict(torch.load("weights/insurance_model.pth"))
model.eval()

def predict_charges(age, sex, bmi, children, smoker, region):
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                              columns=['age','sex','bmi','children','smoker','region'])

    for col in ['sex','smoker','region']:
        input_data[col] = labelencoder[col].transform(input_data[col])

    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    return round(model(input_tensor).item(), 2)
