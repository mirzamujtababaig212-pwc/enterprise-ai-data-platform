FROM python:3.12-slim

LABEL maintainer="Mirza Mujtaba Baig"
LABEL application="Enterprise AI Platform"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENTERPRISE_AI_PLATFORM_ROOT=/app
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

RUN useradd \ 
    --create-home \ 
    --shell /usr/sbin/nologin \ 
    appuser 

RUN apt-get update && \
    apt-get install -y \
        openjdk-21-jdk \
        curl \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY requirements/base.txt requirements/base.txt
COPY requirements/dbt.txt requirements/dbt.txt
COPY requirements/docker.txt requirements/docker.txt

RUN pip install --no-cache-dir --upgrade pip \ 
    && pip install --no-cache-dir -r requirements.txt

RUN pip install \
    --no-cache-dir \
    -r requirements/docker.txt

ENV PATH=$JAVA_HOME/bin:$PATH

COPY . .

RUN chown -R appuser:appuser /app && \
    chmod +x /app/entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uvicorn", "ai_platform.llm_gateway.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
