FROM python:3.12.3-slim
WORKDIR /code
COPY Pipfile /code/Pipfile
COPY bp.keras /code/bp.keras
COPY ./bp_api /code/bp_api
RUN pip install pipenv
RUN pipenv install --deploy
CMD ["pipenv","run","fastapi", "run", "/code/bp_api/bp_api.py", "--port", "80"]

