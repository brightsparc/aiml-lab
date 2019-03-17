import numpy as np
import time
import predict # Import from lambda layer

def lambda_handler(event, context): 
    current_milli_time = lambda: int(round(time.time() * 1000))
    
    # Create a dummy dataset
    input_shape = [1, 3, 112, 112]
    image = np.random.randint(255, size=np.prod(input_shape))
    image = image.reshape(input_shape).astype(int)
    
    # Run the inference
    t1 = current_milli_time()
    output = predict.neo_inference(image)
    response, content_type = predict.neo_postprocess(output)
    t2 = current_milli_time()

    return {
        'status': 200,
        'content_type': content_type,
        'body': response,
        'duration': t2 - t1
    }