### NOTE: this function cannot use MXNet
def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import io

    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'application/vnd+python.numpy+binary':
        raise RuntimeError('Content type must be application/vnd+python.numpy+binary')

    f = io.BytesIO(payload)
    return np.load(f)

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