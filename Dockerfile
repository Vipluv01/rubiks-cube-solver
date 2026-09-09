FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rubiks_solver ./rubiks_solver
COPY results ./results

EXPOSE 8000
CMD ["sh", "-c", "uvicorn rubiks_solver.webapp:app --host 0.0.0.0 --port ${PORT:-8000}"]
