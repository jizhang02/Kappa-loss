# Kappa-loss
This is a loss function based on Kappa index.  
👉 The method has been accepted in [ISBI2020](http://2020.biomedicalimaging.org/).  
👉 The paper can be viewed on [ResearchGate](https://www.researchgate.net/publication/341585606_Kappa_Loss_for_Skin_Lesion_Segmentation_in_Fully_Convolutional_Network).  
👉 Formula:   
![](https://latex.codecogs.com/svg.image?\text{Kappa&space;loss}&space;=&space;1-\frac{2\sum_{i=1}^N{p_ig_i}-\sum_{i=1}^N{p_i}\cdot\sum_{i=1}^N{g_i}/N}{\sum_{i=1}^N&space;{p_i}&plus;\sum_{i=1}^N&space;{g_i}-2\sum_{i=1}^N{p_ig_i}/N})

👉 Code:

```python
import keras
import numpy as np
from keras import backend as K

def Kappa_loss(y_true, y_pred, N=224*224):
    Gi = K.flatten(y_true)
    Pi = K.flatten(y_pred)
    numerator = 2*K.sum(Pi * Gi)-K.sum(Pi)*K.sum(Gi)/N
    denominator = K.sum(Pi*Pi)+K.sum(Gi*Gi)-2*K.sum(Pi*Gi)/N
    Kappa_loss = 1 - numerator/denominator
    return Kappa_loss
 ```
 
### Requirements
* Python 3.*  
* Keras or Tensorflow
* Numpy
### Citation

```
@inproceedings{zhang2020kappa,  
  title={Kappa loss for skin lesion segmentation in fully convolutional network},  
  author={Zhang, Jing and Petitjean, Caroline and Ainouz, Samia},  
  booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},  
  pages={2001--2004},  
  year={2020},  
  organization={IEEE}  
}
```
