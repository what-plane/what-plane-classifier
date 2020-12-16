import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('a350_1.jpg', 'rb')})

print(resp.json())
