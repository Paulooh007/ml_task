FROM python:3.8

WORKDIR /app

COPY app.py ./app
COPY best_model.pkl ./app
COPY requirements.txt ./app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV MODEL_PATH=/app/best_model.pkl

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
