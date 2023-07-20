import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import shutil
import time
import schedule

app = FastAPI()

# Tạo model dữ liệu đầu vào
class InputData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

weights_path = "API_weights.pkl"
with open(weights_path, "rb") as f:
    model = pickle.load(f)

def update_weights():
    source_path = "S_weights.pkl"
    destination_path = "API_weights.pkl"
    shutil.copyfile(source_path, destination_path)
    print("Updated API_weights.pkl")

# Lập lịch chạy vào 5 giờ sáng thứ 2
schedule.every().monday.at("05:00").do(update_weights)
@app.post("/predict")
def predict(data: InputData):
    input_data = pd.DataFrame([data.dict()])
    predictions = model.predict(input_data)
    print(predictions)
    return {"predictions": predictions.tolist()[0]}
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
while True:
    schedule.run_pending()
    time.sleep(86400)