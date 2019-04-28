import boto3
import io
import numpy as np
import json
import time

# Load boto references
s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
sm_runtime = boto3.Session().client('sagemaker-runtime')
iot_client = boto3.client('iot-data')

def load_file(event):
    try:
        # Attempt to load people from s3
        people_object = s3.Object(event['OutputBucket'], event['OutputKey'])
        payload = people_object.get()['Body'].read()
        f = io.BytesIO(payload)
        people = np.load(f)
        return people['vecs'].tolist(), people['names'].tolist(), people['keys'].tolist(), set(people['checksums'])
    except Exception as e:
        print('Initialize new file', e)
        return [], [], [], set()
        
# Save the npz to temp file and get payload
def save_file(event, vecs, names, keys, checksums):
    from tempfile import TemporaryFile
    outfile = TemporaryFile()
    np.savez(outfile, vecs=vecs, names=names, keys=keys, checksums=list(checksums))
    outfile.seek(0)
    payload = outfile.read()
    resp = s3_client.put_object(Bucket=event['OutputBucket'], Key=event['OutputKey'], Body=payload)
    return resp['ETag'].strip('"')    

def get_new_contents(event, checksums, batch_size=10, batch_limit=100):    
    print('getting contents batch size: {}, limit: {}'.format(batch_size, batch_limit))
    contents = []
    is_truncated = False
    
    def filter_by_checksum(response, checksums):
        return [(content['Key'], content['ETag'].strip('"')) for content in response['Contents']
                if content['Size'] > 0 and not content['ETag'].strip('"') in checksums]
    
    # Get the first response
    response = s3_client.list_objects_v2(
        Bucket=event['InputBucket'],
        Prefix=event.get('InputPrefix'),
        MaxKeys=batch_size
    )
    contents += filter_by_checksum(response, checksums)

    # Get remaining response
    while response['IsTruncated']:
        response = s3_client.list_objects_v2(
            ContinuationToken=response['NextContinuationToken'],
            Bucket=event['InputBucket'],
            Prefix=event.get('InputPrefix'),
            MaxKeys=batch_size
        )
        contents += filter_by_checksum(response, checksums)
        if len(contents) > batch_limit:
            print('Reached limit: {}'.format(len(contents)))
            is_truncated = True
            break
    
    return contents, is_truncated

def download_contents(event, contents, vecs, names, keys, checksums):
    for (key, checksum) in contents:
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
        try:
            # Attempt to replace key, or else, append to list
            index = keys.index(key)
            vecs[index] = vec
            names[index] = name
            print('Updated #{} key: {}, name: {}'.format(index, key, name))
        except ValueError:
            print('Added key: {}, name: {}'.format(key, name))
            vecs.append(vec)
            names.append(name)
            keys.append(key)
        checksums.add(checksum)

def update_shadow(thing_name, desired_state):
    shadow = {
        'state': {
            'desired' : desired_state
        }    
    }
    response = iot_client.update_thing_shadow(
        thingName=thing_name,
        payload=json.dumps(shadow)
    )
    shadow = json.loads(response["payload"].read())    
    return shadow['state']['desired']

def function_handler(event, context): 
    current_milli_time = lambda: int(round(time.time() * 1000))

    t1 = current_milli_time()

    ### Load existing vecs names and checksums ###

    vecs, names, keys, checksums = load_file(event)
    print('loaded count: {}'.format(len(checksums)))
    
    ### Get new contents we don't have checksums for ###    
    
    contents, is_truncated = get_new_contents(event, checksums)
    print('Added {} contents, truncated: {}'.format(len(contents), is_truncated))
    
    ### Dowloading contents ###

    print('downloading contents: {}'.format(len(contents)))
    download_contents(event, contents, vecs, names, keys, checksums)
    
    ### Upload new file ###

    people_etag = ''
    
    if len(contents) > 0:
        print('uploading file: {}/{}'.format(event['OutputBucket'], event['OutputKey']))
        people_etag = save_file(event, vecs, names, keys, checksums) 
    
    t2 = current_milli_time()
    
    ### Update shadow ###
    
    if 'ThingName' in event:
        state = {
            'people': {
                'Etag': people_etag,
                'Bucket': event['OutputBucket'],
                'Key': event['OutputKey']
            }
        }
        resp = update_shadow(event['ThingName'], state)
        print('updated shadow', json.dumps(resp))
    
    return {    
        'Added': len(contents),
        'IsTruncated': is_truncated,
        'Total': len(keys),
        'Unique': len(checksums),
        'ETag': people_etag,
        'Duration': t2-t1
    }