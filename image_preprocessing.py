from PIL import Image
import numpy as np

img = Image.open("images/preprocessed/test.png")

img = img.convert("RGB")

img = np.array(img)
print(img)
print(img.shape)
