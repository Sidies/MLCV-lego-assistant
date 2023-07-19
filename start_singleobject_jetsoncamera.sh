#!/bin/sh

cd ./jetson-inference
docker/run.sh --volume ~/AISS_CV_Lego:/AISS -r "python3 ../AISS/frontend/app.py --detection=../AISS/models/Single_Object/Iteration_N_4/lego_Iteration_4.onnx --ssl-key=./data/key.pem --ssl-cert=./data/cert.pem --labels=../AISS/models/Single_Object/Iteration_N_4/lego_Iteration_4_labels.txt --input-layer=input_0 --output-layer=scores,boxes --input=csi://0"


