## Writeup of My Approach to Kaggle Cassava Leaf Classification Competetion [ Private LB 0.8949/Public LB 0.8975 ]

This is a Small writeup of my apporach towards the Cassava Competetion,So,lets discuss some components that I have used:

1. tf_efficientnet_b5_ns from timm 
2. efficientnet_b4 from timm
3. SE-Resnext-50 from timm

### tf_efficientnet_b5_ns 

I have used tf_efficientnet_b5_ns model from timm libaray with wwf(walking with fastai) with this,I have used a albumentation's library with Data Augumentation applied on a image sz of 440x440 the augumentation that I have used on both train and valid set are:

```
def get_train_aug(sz): return albumentations.Compose([
            albumentations.RandomResizedCrop(sz,sz),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5)
])

def get_valid_aug(sz): return albumentations.Compose([
    albumentations.CenterCrop(sz,sz, p=1.),
    albumentations.Resize(sz,sz)
], p=1.)

```
LabelSmoothingCrossEntropy was used as a Loss function,so why I have chosen this because it has shown some awesome results in boosting performance and acc in LB,I have also used GradientAccumulation and ReduceLROnPlateau with it.

The model at the start is submitted solely with 8x TTA(Test Time Augumentation) and able to achive 0.733 in a leaderboard.


### efficientnet_b4

The model has same configration as the above a little chages that I have done is that,I have used some State of the Art Augumentation Techniques MixUP.

#### Configration
Label Smoothing+MixUP+Ranger from Fastai + 15x TTA
Image sz: 440X440

### SE-Resnext-50 from timm

This model is something a game changer which is used by most of the Kagglers in this Competetion.The configration of this model is same as the above no new thing is intriduced while building this.


**At the end all the three models are combined and a average prediction is taken.**
