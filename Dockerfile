FROM python:3.12.2

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt  

EXPOSE 8000

CMD ["chainlit", "run", "app.py"]