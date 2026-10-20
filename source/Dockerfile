FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib HF_HOME=/tmp/huggingface
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd -m -u 1000 appuser
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--no-access-log"]
