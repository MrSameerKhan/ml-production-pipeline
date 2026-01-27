FROM python:3.11-slim

WORKDIR /opt/program

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/app.py /opt/program/app.py
COPY inference/serve /opt/program/serve

RUN chmod +x /opt/program/serve

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

ENTRYPOINT ["/opt/program/serve"]
