# FocusMAE

- This is the official implementation for the CVPR 2024 paper [FocusMAE: Gallbladder Cancer Detection from Ultrasound Videos with Focused Masked Autoencoders](https://arxiv.org/abs/2403.08848).


## DATA PREPARATION

- Refer to the [instructions](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/DATASET.md) in VideoMAE v2 Repository for this step.

- Additionally, we suggest using the `--test_randomization` argument while testing for best results.

- For the region priors using FasterRCNN model, obtain the region proposals in a JSON file for each video using this [code](https://drive.google.com/file/d/1E_LoLKjZ1Co-HrAcPbDasHpDXrJ3Caw2/view).

- Specify the path for folder containing the json files in the dataloader. 


## DATASET

- We contribute additional videos to our Ultrasound video dataset (GBUSV). The complete dataset comprises of 59 videos with malignancy and 32 videos which are benign. The dataset donload instructions are available [in this link.](https://gbc-iitd.github.io/focusmae#dataset)

- The COVID-19 CT Dataset can be obtained [here](https://pubmed.ncbi.nlm.nih.gov/33927208/)

## INSTALLATION 

- The required packages are in the file `requirements.txt`, and you can run the following command to install the environment

```
conda create --name videomae python=3.8 -y
conda activate videomae

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

pip install -r requirements.txt
```

#### Note:
- **The above commands are for reference only**, please configure your own environment according to your needs.
- We recommend installing **`PyTorch >= 1.12.0`**, which may greatly reduce the GPU memory usage.
- It is recommended to install **`timm == 0.4.12`**, because some of the APIs we use are deprecated in the latest version of timm.


## USAGE INSTRUCTIONS

#### Region_Proposals : 

- Obtain the pretrained weights for Faster-RCNN from [here](https://drive.google.com/file/d/1E_LoLKjZ1Co-HrAcPbDasHpDXrJ3Caw2/view).

- Run the below command to obatin the region proposal for your dataset:

```
cd Faster-RCNN
python test.py
```

#### Pre-training : 
- Download the pre-trained model weights for K400 datasets [here](https://github.com/wgcban/adamae/releases/tag/v1).
- Run the below command for pre-training:

```
cd ..
bash pretrain.sh <path_to_kinrtic_400_pretrain_weights> <path_to_output_dir> <path_to_train.csv_file> <path_FRCNN_json_file> <path_to_image_dir>
```

#### Finetuning : 
- Run the below command for finetuning:

```
bash finetune.sh <path_to_checkpoint> <path_to_data_dir>
```

## Acknowledgements
- We thank VideoMAE, VideoMAEv2, and AdaMAE authors for publicly releasing their code. We have built our code-base on top of these fabulous repositories.
