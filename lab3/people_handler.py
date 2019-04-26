import boto3
import io
import numpy as np
import json
import time

# Load boto references
s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
sm_runtime = boto3.Session().client('sagemaker-runtime')

def load_file(event):
    try:
        # Attempt to load people from s3
        people_object = s3.Object(event['OutputBucket'], event['OutputKey'])
        payload = people_object.get()['Body'].read()
        f = io.BytesIO(payload)
        people = np.load(f)
        return people['vecs'].tolist(), people['names'].tolist(), set(people['checksums'])
    except Exception as e:
        print('Initialize new file', e)
        return [], [], set()

# Save the npz to temp file and get payload
def save_file(event, vecs, names, checksums):
    from tempfile import TemporaryFile
    outfile = TemporaryFile()
    np.savez(outfile, vecs=vecs, names=names, checksums=list(checksums))
    outfile.seek(0)
    payload = outfile.read()
    resp = s3_client.put_object(Bucket=event['OutputBucket'], Key=event['OutputKey'], Body=payload)
    return resp['ETag'].strip('"')    
    
def get_new_keys(event, checksums):    
    keys = []
    batch_keys = 10
    limit_keys = 100
    is_truncated = False
    
    def filter_new_keys(response, checksums):
        return [(content['Key'], content['ETag'].strip('"')) for content in response['Contents']
                if content['Size'] > 0 and not content['ETag'].strip('"') in checksums]
    
    # Get the first response
    response = s3_client.list_objects_v2(
        Bucket=event['InputBucket'],
        Prefix=event.get('InputPrefix'),
        Delimiter='/',
        MaxKeys=batch_keys
    )
    keys += filter_new_keys(response, checksums)

    # Get remaining response
    while response['IsTruncated']:
        response = s3_client.list_objects_v2(
            ContinuationToken=response['NextContinuationToken'],
            Bucket=event['InputBucket'],
            Prefix=event.get('InputPrefix'),
            Delimiter='/',
            MaxKeys=batch_keys
        )
        keys += filter_new_keys(response, checksums)
        if len(keys) > limit_keys:
            print('Reached limit: {}'.format(len(keys)))
            is_truncated = True
            break
    
    return keys, is_truncated

def download_keys(event, keys, vecs, names, checksums):
    for (key, checksum) in keys:
        if checksum in checksums:
            print('Skip', checksum)
            continue
        # Get image object
        image_object = s3.Object(event['InputBucket'], key)
        # Get name from meta data or last part of filename
        name = image_object.metadata.get('fullname') or key.split('/')[-1].split('.')[0]
        # Call endpoint to crop boudning box and return vector 
        payload = image_object.get()['Body'].read()
        response = sm_runtime.invoke_endpoint(EndpointName=event['EndpointName'],
                                              ContentType='application/x-image',
                                              Body=payload)
        vec = json.loads(response['Body'].read().decode())
        # Append vector to 
        vecs.append(vec)
        names.append(name)
        checksums.add(checksum)
        print('Added', name)
    
def lambda_handler(event, context): 
    current_milli_time = lambda: int(round(time.time() * 1000))

    t1 = current_milli_time()

    ### Load existing vecs names and checksums ###

    vecs, names, checksums = load_file(event)
    print('loaded count: {}'.format(len(checksums)))    
    
    ### Get new keys we don't have checksums for ###    
    
    keys, is_truncated = get_new_keys(event, checksums)
    print('Added {} keys, truncated: {}'.format(len(keys), is_truncated))

    ### Dowloading keys ###

    print('downloading keys: {}'.format(len(keys)))
    download_keys(event, keys, vecs, names, checksums)    

    ### Upload new file ###

    people_etag = ''
    
    if len(keys) > 0:
        print('uploading file: {}/{}'.format(event['OutputBucket'], event['OutputKey']))
        people_etag = save_file(event, vecs, names, checksums)   
    
    t2 = current_milli_time()
    
    ### Return status ###
    
    return {    
        'Added': len(keys),
        'IsTruncated': is_truncated,
        'Total': len(checksums),
        'ETag': people_etag,
        'Duration': t2-t1
    }