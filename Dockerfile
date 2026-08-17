FROM python:3.12-slim

LABEL maintainer="Mirza Mujtaba Baig"
LABEL application="Enterprise AI Platform"
LABEL version="1.0.0"

USER root

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
COPY requirements/dbt.txt requirements/dbt.txt
COPY requirements/docker.txt requirements/docker.txt

RUN pip install --upgrade pip

RUN pip install \
    --no-cache-dir \
    -r requirements/docker.txt

RUN apt-get update && \
    apt-get install -y \
        openjdk-21-jdk \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]

CMD ["uvicorn", "ai_platform.llm_gateway.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
