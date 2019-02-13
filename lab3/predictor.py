from __future__ import print_function

import os
import base64
import json
import time
import numpy as np
import mxnet as mx
import cv2

from flask import Flask, request, Response

# Include the model name and epoch
prefix = os.getenv('MODEL_PATH', '/opt/ml/')
model_str = os.path.join(prefix, 'mobilenet1,0')
image_size = (112,112)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None # Where we keep the model when it's loaded

    @classmethod
    def get_model(self, ctx=mx.cpu(), layer='fc1'):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.model == None:
            model_parts = model_str.split(',')
            assert len(model_parts)==2
            prefix = model_parts[0]
            epoch = int(model_parts[1])
            print('loading model', prefix, epoch)
            sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
            all_layers = sym.get_internals()
            sym = all_layers[layer+'_output']
            model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
            model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
            model.set_params(arg_params, aux_params)
            self.model = model
        return self.model
    
    @classmethod
    def predict(self, img):
        """For the input, do the predictions and return them.
        Args:
            image: The cv2 image matrix"""
        
        def get_input(img, image_size, bbox=None, rotate=0, margin=0):
            if bbox is None:
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1]*0.0625)
                det[1] = int(img.shape[0]*0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            # Crop
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
            bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
            img = img[bb[1]:bb[3],bb[0]:bb[2],:]
            # Rotate if required
            if 0 < rotate and rotate < 360:
                rows,cols,_ = img.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),360-rotate,1)
                img = cv2.warpAffine(img,M,(cols,rows))
            # Resize and transform
            img = cv2.resize(img, (image_size[1], image_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(img, (2,0,1))
            return aligned

        def l2_normalize(X):
            norms = np.sqrt((X * X).sum(axis=1))
            X /= norms[:, np.newaxis]
            return X

        def get_feature(model, aligned):
            input_blob = np.expand_dims(aligned, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            model.forward(db, is_train=False)
            embedding = model.get_outputs()[0].asnumpy()
            embedding = l2_normalize(embedding).flatten()
            return embedding    
        
        model = self.get_model()
        return get_feature(model, get_input(img, image_size))

# Initialize the Flask application
app = Flask(__name__)
app.debug = True

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return Response(response='{ "status": "ok" }', status=status, mimetype='application/json')

# route http posts to this method
@app.route('/invocations', methods=['POST'])
def test():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """

    print('invocations', request.content_type)
    if request.content_type == 'application/json':
        # convert json base64 encoded data
        body = json.loads(request.data)
        if 'data' not in body:
            raise BadRequestError('Missing image data')
        image = base64.b64decode(body['data']) # byte array
    elif request.content_type == 'application/image-x':
        # convert string of image data to uint8
        image = request.data
    else:
        return Response(response='Require image data', status=415, mimetype='text/plain')

    # Do the prediction    
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    start = time.time()
    predictions = ScoringService.predict(img).tolist()
    duration = time.time()-start
    
    response = json.dumps({ 'predictions': predictions, 'duration': duration })
    return Response(response=response, status=200, mimetype="application/json")

# start flask app
app.run(host="0.0.0.0", port=5000)