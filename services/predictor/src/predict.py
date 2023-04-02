"""
In this script we use the transformed input data and the model to make predicictions.
"""

import os
import numpy as np
import mlflow
from loguru import logger
from fastapi import FastAPI

RUN_ID = os.getenv('RUN_ID', 'f0c63f97bbc74a75aa796be9d729df58')

logged_model = f's3://mlflow-semicon-clf/{RUN_ID}/artifacts/artifacts/'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

app = FastAPI()


@app.post("/predict")
async def predict(row: dict):
    # have to convert to int because numpy is not supported by fastapi : https://github.com/tiangolo/fastapi/issues/2293
    row = np.array(list(row.values())).reshape(1, -1) 
    pred = int(loaded_model.predict(row))
    return {'prediction': pred, 'model_version': RUN_ID}


if __name__ == "__main__":
    # Use this for debugging perpuses
    logger.debug("Running in developement mode. Do not run like this in production")

    import uvicorn
    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
