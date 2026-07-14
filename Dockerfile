FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app
RUN pip isnatll -r requirements.txt
CMD ["python3","app.py"]