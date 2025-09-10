FROM python:3.12-slim

ENV SCRIPT_TO_RUN=train

WORKDIR /app

COPY . /app

# Instalamos libgomp1 para que LightGBM funcione
RUN apt-get update && apt-get install -y libgomp1

# Instalamos las librerías Python
RUN pip install -r requirements.txt

# NO CAMBIAR ESTAS RUTAS
ENV INFERENCE_DATA_PATH=/app/bookings_test.csv
ENV HOTELS_PATH=/app/hotels.csv
ENV BOOKINGS_PATH=/app/bookings_train.csv
ENV MODEL_PATH=/app/pipeline.cloudpkl

CMD ["sh", "-c", "python -m $SCRIPT_TO_RUN"]

# PARA CREAR LA IMAGEN
# docker build . -t uax-entrega:latest

# PARA CORRER EL CONTENEDOR DE ENTRENAMIENTO
# docker run -d -v .:/app -e SCRIPT_TO_RUN=train uax-entrega:latest

# PARA CORRER EL CONTENEDOR DE INFERENCIA
# docker run -d -v .:/app -e SCRIPT_TO_RUN=inference uax-entrega:latest
