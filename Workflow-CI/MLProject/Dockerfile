FROM python:3.8-slim

WORKDIR /app

COPY conda.yaml .
COPY modelling.py .
COPY dataset_preprocessing/ ./dataset_preprocessing/

RUN pip install mlflow scikit-learn pandas numpy matplotlib seaborn

CMD ["python", "modelling.py"]