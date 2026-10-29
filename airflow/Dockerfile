FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
USER airflow

COPY airflow/requirements.txt /tmp/airflow-reqs.txt
RUN pip install --no-cache-dir -r /tmp/airflow-reqs.txt

# make src/ importable inside the container
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"
