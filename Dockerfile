FROM python:3.9-slim-buster

COPY . /agribrain-api/

WORKDIR /agribrain-api/

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

EXPOSE 8085

ENTRYPOINT ["python", "api.py"]
