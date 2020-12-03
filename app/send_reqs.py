import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('6216841.jpg', 'rb')})

print(resp.json())
