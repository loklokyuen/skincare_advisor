FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY .streamlit .streamlit
COPY app app
COPY config config
COPY graph graph
COPY services services
COPY tools tools
COPY utils utils
COPY schemas.py .

EXPOSE 8080

CMD ["streamlit", "run", "app/Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
