FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3

WORKDIR /app


COPY ../requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


CMD ["python", "./src/test.py"]