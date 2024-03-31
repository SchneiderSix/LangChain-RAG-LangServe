import requests

response = requests.post(
    "http://localhost:8000/qp/invoke",
    json={"input": {
        "question": "what can you tell me about promptior?"}
    }
)

try:
    data = response.json()
    print(data['output'])
except Exception as e:
    print("Error decoding JSON:", e)
    print("Response Text:", response.text)
