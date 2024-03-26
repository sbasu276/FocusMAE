## FocusMAE

This is the official implementation for the CVPR 2024 paper [FocusMAE: Gallbladder Cancer Detection from Ultrasound Videos with Focused Masked Autoencoders](https://arxiv.org/abs/2403.08848).


[CVPR Weigths](https://drive.google.com/drive/folders/16E1EDl323GFAbmQ02fqVQwVkkz-4GBZY?usp=sharing)

## DATA PREPARATION

Refer to the [instructions](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/DATASET.md) in VideoMAE v2 Repository for this step.

Additionally, we suggest using the --test_randomization argument while testing for best results.


## INSTALLATION 

The required packages are in the file `requirements.txt`, and you can run the following command to install the environment

```
conda create --name videomae python=3.8 -y
conda activate videomae

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

pip install -r requirements.txt
```

### Note:
- **The above commands are for reference only**, please configure your own environment according to your needs.
- We recommend installing **`PyTorch >= 1.12.0`**, which may greatly reduce the GPU memory usage.
- It is recommended to install **`timm == 0.4.12`**, because some of the APIs we use are deprecated in the latest version of timm.


## Training and Testing Scripts

The folder scripts contains files for [Finetuning](scripts/finetune_train.sh) and [Pre-training](scripts/pretrain_train.sh).

In each script specify the following:
#### OUTPUT_DIR : 
- Working directory name which saves all the checkpoints
- Each working directory folder structure looks like this:- Dataset_folder/work_dir/output_dir_name/checkpoint_folder
- Download the model checkpoints and pretrained folders in the same format

#### MODEL_PATH : 
- Specify the path of the pretrained model to finetune from 
- You can download the pretrained models and arrange then in the folder structure shown above.



Additionally, we provide our training and testing scripts as examples which can be used as follows 
```bash scripts/finetune_train.sh``` 
