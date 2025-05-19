#!/bin/bash

exp=$1
echo "exp: $exp"
apptainer exec -H $pwd:/home --nv --mount type=bind,src=<data-folder>,dst=/data pytorch_2.3.1-cuda12.1-cudnn8-devel.sif /bin/bash -c "cd /home && ./exp/$exp.sh"
