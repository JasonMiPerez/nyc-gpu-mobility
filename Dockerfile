FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8050
COPY pyproject.toml /app/
COPY src /app/src
RUN pip install --no-cache-dir .
COPY app.py /app/
RUN useradd --create-home --uid 10001 mobility
USER mobility
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/healthz')"
# Mount one validated published snapshot read-only at /snapshot.
ENV MOBILITY_SNAPSHOT=/snapshot
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "2", "--timeout", "60", "--access-logfile", "-"]
