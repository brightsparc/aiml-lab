import requests
import base64
import json

# Define target url
url = 'http://localhost:8080/invocations'

# Load sample image
data = open('Tom_Hanks_54745.png', 'rb').read()

# Post image data directly to endpoint
response = requests.post(url, data=data, headers={'Content-Type': 'application/image-x'})
print('application/image-x', response.json())

# Wrap data in base64 encoded json, and pass down specific bounding box
json_data = json.dumps({ 
    'data': base64.b64encode(data).decode("utf-8", "ignore"),
    'bbox': [1, -3, 84, 118, 0]
})
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})
print('application/json', response.json())
