#!/bin/bash

export URL='http://localhost:8080/invocations'
export PIC='Tom_Hanks_54745.png'
(echo -n '{"data": "'; base64 $PIC; echo '", "bbox": [1, -3, 84, 118, 0] }') | curl -H "Content-Type: application/json" -d @- $URL
