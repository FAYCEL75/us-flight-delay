# ============================
# Dockerfile.app — Streamlit
# ============================
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY config.yml ./config.yml

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

ENV PYTHONPATH="/app"

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]