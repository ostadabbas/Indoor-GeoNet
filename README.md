# IndoorGeoNet

This code is the implementation of the following paper in Tensorflow:

IndoorGeoNet: Weakly Supervised Hybrid Learning for Depth and Pose Estimation

Amirreza Farnoosh and Sarah Ostadabbas


## Requirements

This code is tested on Python3.6, TensorFlow 1.1 and CUDA 8.0 on Ubuntu 16.04.

## Data preparation

The following datasets are used for experiments in the paper:

### [RSM Hallway](http://www.bicv.org/datasets/rsm-dataset/)
### [MobileRGBD](http://mobilergbd.inrialpes.fr/)

You should use the following command to preprocess dataset:
```bash
python data/prepare_train_data.py --dataset_dir=/path/to/dataset/ --dataset_name= data_name --dump_root=/path/to/formatted/data/ --seq_length=5 --img_height=144 --img_width=256
```

For **RSM Hallway** dataset, the `--dataset_name` should be `rms_hallway`, and for **MobileRGBD** dataset, the `--dataset_name` should be `mobileRGBD`;

## Training

You should run the following command for training the network: 

```bash
python geonet_main.py --mode=train_rigid --dataset_dir=/path/to/formatted/data/ --checkpoint_dir=/path/to/save/ckpts/ --learning_rate=0.0002 --seq_length=5 --batch_size=4 --max_steps=150000 
```

## Testing

### Monocular Depth
Run the following command for depth predictions:
```bash
python geonet_main.py --mode=test_depth --dataset_dir=/path/to/raw/dataset/ --pose_test_seq= test_folder_name --init_ckpt_file=/path/to/trained/model/ --batch_size=1 --output_dir=/path/to/save/predictions/ --dataset_name= data_name
```

### Camera Pose
Run the following command for pose predictions:
```bash
python geonet_main.py --mode=test_pose --dataset_dir=/path/to/raw/dataset/ --depth_test_seq= test_folder_name --init_ckpt_file=/path/to/trained/model/ --batch_size=1 --seq_length=5 --output_dir=/path/to/save/predictions/ --dataset_name= data_name
```

## Reference

```
@inproceedings{amir2018indoorgeonet,
  title     = {Weakly Supervised Hybrid Learning for Depth and Pose Estimation},
  author    = {Farnoosh, Amirreza and Ostadabbas, Sarah},
  booktitle = {arxiv},
  year = {2018}
}
```
