FROM python:3.8-slim

WORKDIR /app

# Copy just the requirements first to leverage Docker cache
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your app's code
COPY app.py .

EXPOSE 8080

CMD ["python", "./app.py"]
