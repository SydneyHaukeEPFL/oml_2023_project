# Requirements
- torch
- torchmetrics
- numpy

# Usage
A model can be trained running the `main.py` file:
```{bash}
python main.py
    --dataset=[wine, Cifar10, Cifar100]
    --model=[mlp, simple_cnn, resnet18, resnet50]
    
    --optimizer=[sgd, zero_order]
    --lr=(float)
    --u=(float)
    --batch_size=(int)
```
The results and the final model weights are stored in `/results`.

# Reproducibility
The script that we used to train our models are `tunining_cifar.sh` and `tuning_wine.sh`.

Then the analysis of the results are done using the two jupyter notebooks: `analysis_cifar10.ipynb` and `analysis_wine.ipynb`
