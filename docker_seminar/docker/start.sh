#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

if command -v nvidia-smi &> /dev/null
then
    echo "Running on ${orange}nvidia${reset_color} hardware"
    ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
else
    echo "Running on ${orange}intel${reset_color} hardware: nvidia driver not found"
    ARGS="--device=/dev/dri:/dev/dri"
fi

docker run -it \
    $ARGS \
    --ipc host \
    --privileged \
    -p ${UID}0:22 \
    --name mirea \
    segmentator:latest

docker exec --user root \
    mirea bash -c "/etc/init.d/ssh start"
