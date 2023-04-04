FROM python:3.9-slim-buster

COPY . /agribrain-api/

WORKDIR /agribrain-api/

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "api.py"]