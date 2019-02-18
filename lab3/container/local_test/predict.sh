#!/bin/bash

export URL='http://localhost:8080/invocations'
export PIC='../tmp/image'
(echo -n '{"data": "'; base64 $PIC; echo '", "topk": 3}') | curl -H "Content-Type: application/json" -d @- $URL
