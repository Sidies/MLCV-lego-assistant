#!/bin/sh

cd ./jetson-inference
docker/run.sh --volume ~/AISS_CV_Lego:/AISS -r "python3 ../AISS/frontend/app.py --detection=../AISS/models/custom/model.onnx --ssl-key=./data/key.pem --ssl-cert=./data/cert.pem --labels=../AISS/models/custom/model_labels.txt --input-layer=input_0 --output-layer=scores,boxes --input=csi://0"


