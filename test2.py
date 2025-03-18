import requests
import json

url = "http://127.0.0.1:5000/predict"
data = {"name": "1921", "step": 10}
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers)
with open('prediction_plot.png', 'wb') as f:
    f.write(response.content)
    print("图像已保存为 prediction_plot.png")