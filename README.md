# Zero-shot versus Many-shot: Unsupervised Texture Anomaly Detection

This repository contains the code for the above mentioned paper submission:

Sample results using zero-shot approach on MVTec AD datasets: 

![example](mvtec_zeroshot.png "Sample image restoration results")

## Requirement
* Python version 3.7
* PyTorch version 1.9.1
* Torchvision version 0.10.1


## Usage

### Run 
```shell
python main.py
```

#### Dataset
The MVTec AD dataset is used in this code. When run, the dataset will be downloaded automatically if it is not found in the data directory.
When the download is completed, the main code for the proposed method will be executed. Note that training data is not used since this is a zero-shot approach.

To use other dataset, please modify the code in `~/src/datasets/mvtec.py`.