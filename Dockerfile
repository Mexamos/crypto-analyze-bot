FROM python:3.11-slim

COPY . /src
WORKDIR /src

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
