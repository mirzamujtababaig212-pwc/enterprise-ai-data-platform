FROM apache/spark:3.5.6
USER root
WORKDIR /app
COPY requirements/docker.txt requirements/docker.txt
COPY requirements/base.txt requirements/base.txt
COPY requirements/dbt.txt requirements/dbt.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements/docker.txt
COPY . .
ENV PYTHONPATH=/app
CMD ["pytest"]
