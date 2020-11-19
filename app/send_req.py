import requests
import os

image_dir = 'test_images/'
files = os.listdir(image_dir)

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open(image_dir+'90867.jpg', 'rb')})
print(resp.json())
# for file in files:
#     resp = requests.post("http://localhost:5000/predict",
#                          files={"file": open(image_dir+file, 'rb')})
#     print((file, resp.json()))
