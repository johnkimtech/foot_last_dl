FROM  nvcr.io/nvidia/pytorch:23.09-py3 AS footlastdl_build
WORKDIR /app
RUN apt update
RUN apt-get install -y libx11-6 libgl-dev
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM footlastdl_build
WORKDIR /app
COPY . /app
EXPOSE 7860
CMD ["python", "app.py"]