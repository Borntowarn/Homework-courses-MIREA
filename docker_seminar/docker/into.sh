#!/bin/bash

docker exec --user "docker_user" -it segmentator \
        /bin/bash -c "cd /home/${USER}; /bin/bash"
