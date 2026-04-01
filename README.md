# CPENet

This repo is an official implementation of the *(Collaborative Prior-Enhanced RGB-D Salient Object Detection Network for Intelligent IoT Perception Devices)CPENet*.

## Prerequisites

## Usage

### 1. Clone the repository

### 2. Prepare the data 
Use the erosion_dilate.py to generate the region masks for region loss.

### 3. Training

You can train the model by using 
```
python train.py
```

### 4. Testing
```
python test.py
```

### 5. Evaluation

We provide [saliency maps](https://pan.baidu.com/s/17mNaTgz7dDZUQ_gv_9RZqg?pwd=2671) (fetch code: 2671) of our CPENet on seven datasets.
### 6. citation

@article{gao2026collaborative,
  title={Collaborative Prior-Enhanced RGB-D Salient Object Detection Network for Intelligent IoT Perception Devices},
  author={Gao, Lina and Chen, Haikun and Zhang, Yonggang and Huang, Yulong},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
