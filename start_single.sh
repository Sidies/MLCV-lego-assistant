#!/bin/sh

cd ./jetson-inference
docker/run.sh --volume ~/AISS_CV_Lego:/AISS -r "python3 ../AISS/frontend/app.py --detection=../AISS/models/Single_Object/Iteration_6/lego_Iteration_6.onnx --ssl-key=./data/key.pem --ssl-cert=./data/cert.pem --labels=../AISS/models/Single_Object/Iteration_6/lego_Iteration_6_labels.txt --input-layer=input_0 --output-layer=scores,boxes"


