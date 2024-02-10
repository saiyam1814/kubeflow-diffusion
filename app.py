from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, safety_checker=None)

@app.route("/v1/models/serving:predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data["prompt"]
    image = pipe(prompt).images[0]  # Generate the image
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

if __name__ == "__main__":
    http_server = WSGIServer(("", 8080), app)
    http_server.serve_forever()

