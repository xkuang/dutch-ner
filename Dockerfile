FROM python:3.6

EXPOSE 5000

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt
CMD python index.py