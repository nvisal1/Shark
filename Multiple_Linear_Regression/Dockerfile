FROM continuumio/anaconda3

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "rest_api.py"]