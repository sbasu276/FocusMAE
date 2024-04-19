# FocusMAE

This is the official implementation for the CVPR 2024 paper [FocusMAE: Gallbladder Cancer Detection from Ultrasound Videos with Focused Masked Autoencoders](https://arxiv.org/abs/2403.08848).


## DATA PREPARATION

Refer to the [instructions](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/DATASET.md) in VideoMAE v2 Repository for this step.

Additionally, we suggest using the `--test_randomization` argument while testing for best results.

For the region priors using FasterRCNN model, obtain the region proposals in a JSON file for each video using this [code](https://drive.google.com/file/d/1E_LoLKjZ1Co-HrAcPbDasHpDXrJ3Caw2/view).

Specify the path for folder containing the json files in the dataloader. 


## DATASET

We contribute additional videos to our Ultrasound video dataset (GBUSV). The complete dataset comprises of 59 videos with malignancy and 32 videos which are benign. The dataset donload instructions are available [in this link.](https://gbc-iitd.github.io/focusmae#dataset)

The COVID-19 CT Dataset can be obtained [here](https://pubmed.ncbi.nlm.nih.gov/33927208/)

## INSTALLATION 

The required packages are in the file `requirements.txt`, and you can run the following command to install the environment

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

The folder scripts contains files for [Finetuning](scripts/finetune_train.sh) and [Pre-training](scripts/pretrain_train.sh).

In each script specify the following:
#### OUTPUT_DIR : 
- Working directory name which saves all the checkpoints
- Each working directory folder structure looks like this:- 
```Dataset_folder/work_dir/output_dir_name/checkpoint_folder```
- You can either download the model checkpoints and pretrained folders in the same format, or download individual checkpoint from the links in the table and place them in the folder structure desscribed above.


#### MODEL_PATH : 
- Specify the path of the pretrained model to finetune from 
- You can download the pretrained models and arrange then in the folder structure shown above.

Our pretrained models and checkpoints can be downloaded from this link : [CVPR Weigths](https://drive.google.com/drive/folders/16E1EDl323GFAbmQ02fqVQwVkkz-4GBZY?usp=sharing)

| Model Name                       | Link                         |
|----------------------------------|------------------------------|
| Pre-trained model for GBC Dataset  | https://tinyurl.com/3s6567c3 | 
| Finetuning ckpt - Fold_0 GBC dataset | https://tinyurl.com/4y2phujr |
| Finetuning ckpt - Fold_1 GBC dataset | https://tinyurl.com/ajazhb79 |
| Finetuning ckpt - Fold_2 GBC dataset | https://tinyurl.com/3jptv2dp |
| Finetuning ckpt - Fold_3 GBC dataset | https://tinyurl.com/2r9ywuzj |
| Finetuning ckpt - Fold_4 GBC dataset | https://tinyurl.com/25zuures |
| Pretrained model for CT Dataset | [here](https://drive.google.com/file/d/1G2BzBzOARGyeam2B-zVuUN0KphhkMeAe/view?usp=sharing) |
| Finetuning ckpt - CT Dataset | [here](https://drive.google.com/file/d/14eP3hx9M3E5HM0GZvp2QbUgu86DMPNU3/view?usp=sharing) |


Additionally, we provide our training and testing scripts as examples which can be used as follows 
```bash scripts/finetune_train.sh``` 

## Acknowledgements
We thank VideoMAE, VideoMAEv2, and AdaMAE authors for publicly releasing their code. We have built our code-base on top of these fabulous repositories.
