import numpy as np
import time
import base64
import predict # Import from lambda layer

model = None

def lambda_handler(event, context): 
    global model
    
    current_milli_time = lambda: int(round(time.time() * 1000))
    
    # Decode the payload
    payload = base64.b64decode(event['Image']['Bytes'])
    bbox = event['BoundingBox']
    image = predict.neo_preprocess(payload, 'application/x-image', bbox=bbox)
    
    # Run the inference
    t1 = current_milli_time()
    if model == None:
        model = predict.neo_load()
        t2 = current_milli_time()
        print('loaded model in {}ms'.format(t2-t1))
    t3 = current_milli_time()
    output = predict.neo_inference(model, image)
    t4 = current_milli_time()
    print('model inference in {}ms'.format(t4-t3))
    response, content_type = predict.neo_postprocess(output)
    t5 = current_milli_time()

    return {
        'Status': 200,
        'ContentType': content_type,
        'Body': response,
        'Duration': t5 - t1
    }