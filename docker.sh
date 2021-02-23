docker run \
    -it \
    --gpus all \
    -v $HOME:$HOME \
    -v $HOME/.cache:/root/.cache \
    -w $PWD \
    tensorrt:7.1 /bin/bash
