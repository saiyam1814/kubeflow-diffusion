from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import base64
from io import BytesIO

app = Flask(_name_)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, safety_checker=None,)

@app.route("/v1/models/serving:predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data["prompt"]
    # image = pipe(prompt).images[0]  # Generate the image
    vae = pipe.vae
    images = []

    def latents_callback(i, t, latents):
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        images.extend(pipe.numpy_to_pil(image))

    def image_grid(imgs, rows, cols):
        w,h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid

    torch.manual_seed(9000)
    final_image = pipe(prompt, callback=latents_callback, callback_steps=1, num_inference_steps=5).images[0]
    images.append(final_image)

    buffered = BytesIO()
    image_grid(images, rows=5, cols=len(images)).save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

if _name_ == "_main_":
    http_server = WSGIServer(("", 8080), app)
    http_server.serve_forever()




