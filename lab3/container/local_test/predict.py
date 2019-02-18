import requests
import base64

# Post image data directly to endpoint
data = open('../tmp/image', 'rb').read()
response = requests.post('http://localhost:8080/invocations', 
    data=data, headers={'Content-Type': 'application/image-x'})
print('application/image-x', response.json())

# Wrap data in base64 encoded json
json_data = json.dumps({ 'data': base64.b64encode(data).decode("utf-8", "ignore") })
response = requests.post('http://localhost:5000/invocations', data=json_data, headers={'Content-Type': 'application/json'})
print('application/json', response.json())
