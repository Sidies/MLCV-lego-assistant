#!/bin/sh

cd ./jetson-inference
docker/run.sh --volume ~/AISS_CV_Lego:/AISS -r "python3 ../AISS/frontend/app.py --detection=../AISS/models/Combined_Object/Iteration_7/lego_Iteration_7.onnx --ssl-key=./data/key.pem --ssl-cert=./data/cert.pem --labels=../AISS/models/Combined_Object/Iteration_7/lego_Iteration_7_labels.txt --input-layer=input_0 --output-layer=scores,boxes"


