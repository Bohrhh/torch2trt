docker run \
    -it \
    --gpus all \
    -v $HOME:$HOME \
    -w $PWD \
    tensorrt:7.1 /bin/bash
