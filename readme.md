
# AI4Good - iNaturalist 2018 Challenge 
Code adapted from [here](https://github.com/macaodha/inat_comp_2018).

## Branches
- master: train torchvision models (resnet)
- vit: train vision transformer

## Trained models
Download trained baseline models [from polybox](https://polybox.ethz.ch/index.php/s/10yX4iEPP9caOog)

## Installation
```
conda create -n ai python=3.8
conda activate ai
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install timm
pip install matplotlib
```
