docker run \
    -it \
    --gpus all \
    -v $HOME:$HOME \
    -v /mnt/38BE24F1BE24A8F8:/mnt/38BE24F1BE24A8F8 \
    -v /mnt/4ACC22D0CC22B65B:/mnt/4ACC22D0CC22B65B \
    -v /mnt/Fly:/mnt/Fly \
    -v /mnt/Fighting:/mnt/Fighting \
    -v $HOME/.cache:/root/.cache \
    -w $PWD \
    tensorrt:7.1 /bin/bash
