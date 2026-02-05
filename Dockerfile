FROM python:3.10-slim-buster
WORKDIR /app
COPY . .
Run pip install -r requirement.txt

CMD ["python3","app.py"]