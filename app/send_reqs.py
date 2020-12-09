import requests

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('a350.jpeg', 'rb')})

print(resp.json())
