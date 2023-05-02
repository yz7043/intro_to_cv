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
create folders for recolor tasks

```shell
mkdir edge2rgb
cd edge2rgb
mkdir trainA
mkdir trainB
```

```shell
cd checkpoints
mkdir edge2rgb
```

```shell
cd save_images
mkdir edge2rgb
cd edge2rgb
mkdir fake_color
mkdir fake_edge
```

```Shell
cp horse2zebra/trainA/* edge2rgb/trainB/
cp horse2zebra/trainB/* edge2rgb/trainB/
python getEdgeImage.py
```



## How to set training parameters

You need to modify file inside config.py

## How to train a model
```shell
# Train the horse-to-zebra model
python train.py --is_h2z True --tqdm False
# Train the edge-recolor model
python train.py --is_color True --tqdm False
```
if there is any package missing, install them.