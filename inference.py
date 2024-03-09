import requests
from IPython.display import display
from PIL import Image
import base64
from io import BytesIO

response = requests.post(
    f"URL",
    json={"prompt": ""},
)

data = response.json()
img_data = base64.b64decode(data["image"])
img = Image.open(BytesIO(img_data))
display(img)
