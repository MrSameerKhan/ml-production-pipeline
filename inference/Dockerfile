FROM python:3.11-slim

WORKDIR /opt/program

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/app.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "app.py"]
