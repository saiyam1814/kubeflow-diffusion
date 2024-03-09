from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# Load the model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, safety_checker=None)

# Define the predict route
@app.route("/v1/models/serving:predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data["prompt"]
    images = []

    # Define the callback for processing latent images
    def latents_callback(i, t, latents):
        if i < 5:  # Capture only the first five latents
            modified_latents = 1 / 0.18215 * latents  # Modify latents if necessary
            image = pipe.vae.decode(modified_latents).sample[0]
            image = (image / 2 + 0.5).clamp(0, 1)  # Normalize image
            image = image.cpu().permute(1, 2, 0).numpy()  # Convert to NumPy
            pil_image = Image.fromarray((image * 255).astype('uint8'))  # Convert to PIL Image
            images.append(pil_image)

    # Generate the final image
    torch.manual_seed(9000)  # For reproducibility
    pipe(prompt, callback=latents_callback, callback_steps=1, num_inference_steps=5)
    final_image = pipe(prompt, num_inference_steps=5).images[0]  # Redundant generation for clear final image
    images.append(final_image)  # Append the final clear image

    # Function to create an image grid
    def image_grid(imgs, rows, cols):
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    # Create the grid and encode the image to base64
    buffered = BytesIO()
    grid = image_grid(images, rows=len(images) // 5 + (1 if len(images) % 5 > 0 else 0), cols=5)
    grid.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

# Run the server
if __name__ == "__main__":
    http_server = WSGIServer(("0.0.0.0", 8080), app)
    http_server.serve_forever()

