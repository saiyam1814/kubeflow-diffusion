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

# Consider where and how you'll be using this model file in your app
RUN wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt -O model.ckpt

EXPOSE 8080

CMD ["python", "./app.py"]
