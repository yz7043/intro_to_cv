## How to set up project before training
create folders holding checkpoint-models
```shell
mkdir checkpoints  
cd checkpoints  
mkdir horse2zebra  
```

download the dataset
```shell
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip  
unzip horse2zebra.zip
```

create folders for saving pictures
```shell
mkdir save_images
cd save_images
mkdir horse2zebra
cd horse2zebra
mkdir fake_horse
mkdir fake_zebra
```

## How to set training parameters
You need to modify file inside config.py