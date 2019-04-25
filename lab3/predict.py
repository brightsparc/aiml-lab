def model_fn(model_dir, prefered_batch_size=1, image_size=(112,112)):
    """Function responsible for loading the model.
    Args:
        model_dir (str): The directory where model files are stored
    Returns:
        mxnet.mod.Module: the loaded model.
    """
    import platform
    import mxnet as mx
    import os
    import logging
    
    import subprocess
    import sys
    
    # Workaround until support requirements.txt for dependencies
    # See: https://github.com/aws/sagemaker-python-sdk/issues/284
    logging.info('Installing dependencies for pillow')
    
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pillow'])
                             
    logging.info('Invoking model load for py:{} mxnet:{}'.format(
        platform.python_version(), mx.__version__))
        
    data_shapes = [('data', (prefered_batch_size, 3, image_size[0], image_size[1]))]

    sym, args, aux = mx.model.load_checkpoint(os.path.join(model_dir, 'model'), 0)

    ctx = mx.cpu()
                   
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(for_training=False, data_shapes=data_shapes)
    model.set_params(args, aux, allow_missing=True)
    
    return model

def transform_fn(model, request_body, request_content_type, accept_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param model: The model.
    :param request_body: The request payload.
    :param request_content_type: The request content type.
    :param accept_type: The (desired) response content type.
    :return: response payload and content type.
    """

    import mxnet as mx
    import logging
    
    logging.info('Invoking transform Shape: {}, Content Type: {}, Accept: {}'.format(
        model.data_shapes[0][1], request_content_type, accept_type))
    
    array = neo_preprocess(request_body, request_content_type)
    
    logging.debug('Model input: {}'.format(array))

    data = mx.nd.array(array)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    output = model.get_outputs()[0].asnumpy()

    logging.debug('Model output: {}'.format(output))

    if accept_type == 'application/json':
        return neo_postprocess(output)
    else:
        raise RuntimeError('Accept header must be application/json')

### NOTE: This function is used within lambda layer
def neo_load(model_path='compiled', device='cpu'):
    import logging
    import os
    from dlr import DLRModel # Import the relative dlr library
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)    
    logging.info('Loading model: {} on: {}'.format(model_path, device))
    
    return DLRModel(model_path, device)

def neo_inference(model, data):
    import logging

    logging.info('Invoking inference for model')
    
    input_data = {'data': data}
    
    return model.run(input_data)[0]

### NOTE: this function cannot use MXNet
def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io
    
    def get_bbox_roll(payload):
        # Only import boto3 if we need to call rekognition API for bbox/roll
        import boto3
        logging.debug('calling rekognition')                 
        rekognition = boto3.client('rekognition')
        ret = rekognition.detect_faces(
            Image={ 'Bytes': payload },
            Attributes=['DEFAULT'],
        )
        # Return bbox and roll rounded to 90 degrees
        return ret['FaceDetails'][0]['BoundingBox'], round(ret['FaceDetails'][0]['Pose']['Roll']/90)*90
    
    def crop_image(payload, bbox=None, roll=0, margin=0, image_size=(112, 112)):
        # Only load PIL if required to transform bytes
        import PIL.Image
        logging.debug('crop image bbox: {}, roll: {}, margin: {}'.format(bbox, roll, margin))                 
        # Load image and convert to RGB space
        f = io.BytesIO(payload)
        image = PIL.Image.open(f).convert('RGB')
        # Crop relative to image size
        if bbox != None:
            width, height = image.size
            x1 = int(bbox['Left'] * width)
            y1 = int(bbox['Top'] * height)
            x2 = int(bbox['Left'] * width + bbox['Width'] * width)
            y2 = int(bbox['Top'] * height + bbox['Height']  * height)
            # Add margin as 
            if margin > 0:
                if isinstance(margin, float):
                    margin = int(margin*width)
                x1 = np.maximum(x1-margin, 0)
                y1 = np.maximum(y1-margin, 0)
                x2 = np.minimum(x2+margin, width)
                y2 = np.minimum(y2+margin, height)
            image = image.crop((x1, y1, x2, y2))
        # Rotate expanding size
        if roll != 0:
            image = image.rotate(roll, expand=True)        
        # Resize
        image = image.resize(image_size)
         # Transpose
        return np.rollaxis(np.asarray(image), axis=2, start=0)[np.newaxis, :]   

    logging.info('Invoking user-defined pre-processing function')

    if content_type == 'application/x-npy':
        f = io.BytesIO(payload)
        return np.load(f)
    elif content_type == 'application/x-image':
        # Get bbox if we have image only
        bbox, roll = get_bbox_roll(payload)
        return crop_image(payload, bbox, roll)
    elif content_type == 'application/json':
        import json
        import base64
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode('utf-8')
        event = json.loads(payload)
        if not 'Image' in event and not 'Bytes' in event['Image']:
            raise RuntimeError('Require Image Bytes for application/json')
        # Decode base64 image bytes
        payload = base64.b64decode(event['Image']['Bytes'])
        # If we have bounding box and roll pass these through
        if 'BoundingBox' in event and 'Roll' in event:
            bbox, roll = event['BoundingBox'], event['Roll']
        else:
            bbox, roll = get_bbox_roll(payload)
        return crop_image(payload, bbox, roll)
    else:
        raise RuntimeError('Content type must be application/json, application/x-image or application/x-npy')

### NOTE: this function cannot use MXNet
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    def l2_normalize(X):
        norms = np.sqrt((X * X).sum(axis=1))
        X /= norms[:, np.newaxis]
        return X

    logging.info('Invoking user-defined post-processing function')

    # Return a normalize embedding
    embedding = l2_normalize(np.array(result)).flatten()
    
    response_body = json.dumps(embedding.tolist())
    content_type = 'application/json'

    return response_body, content_type