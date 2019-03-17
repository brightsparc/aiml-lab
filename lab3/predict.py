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
    
    print('input', array)

    data = mx.nd.array(array)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    output = model.get_outputs()[0].asnumpy()
    
    print('output', output)

    if accept_type == 'application/json':
        return neo_postprocess(output)
    else:
        raise RuntimeError('Accept header must be application/json')

### NOTE: This function is used within lambda layer        
def neo_inference(data, model_path='compiled', device='cpu'):
    import logging
    import os
    from dlr import DLRModel # Import the relative dlr library
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)    
    logging.info('Invoking inference for model: {} on: {}'.format(model_path, device))
    
    model = DLRModel(model_path, device)
    input_data = {'data': data}
    return model.run(input_data)[0]

### NOTE: this function cannot use MXNet
def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io

    logging.info('Invoking user-defined pre-processing function')

    if content_type == 'application/x-npy':
        f = io.BytesIO(payload)
        return np.load(f)
    elif content_type == 'application/x-image':
        import PIL.Image # Training container doesn't have this package
        f = io.BytesIO(payload)
        image = PIL.Image.open(f).convert('RGB') # Load image and convert to RGB space
        image = np.asarray(image.resize((112, 112))) # Resize
        image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :] # Transpose
    else:
        raise RuntimeError('Content type must be application/x-image or application/x-npy')

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
    
    print('embedding', embedding)
    
    response_body = json.dumps(embedding.tolist())
    content_type = 'application/json'

    return response_body, content_type