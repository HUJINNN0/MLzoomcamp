FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv
RUN pip install numpy
RUN pip install scipy

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress", "--bind=0.0.0.0:9696", "predict:app"]