FROM python:3.13-slim
RUN pip install pipenv
WORKDIR /app                                                                
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system
COPY ["*.py", "dv.bin", "model1.bin", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "model:app"]