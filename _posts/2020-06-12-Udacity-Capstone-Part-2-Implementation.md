---
toc: false
layout: post
description: A Udacity Machine Learning Engineer Nanodegree Capstone Project
categories: [deep learning, fastai2, audio, markdown]
title: Multi-Label Auto-Tagging of Noisy Audio Using fastai2 - Part 2 - Methodology (Data Pre-processing, Implementation and Refinement)
---


Welcome to Part 2 of a blog series based on my Udacity Machine Learning Engineer Nanodegree Capstone project.  This section defines the model implementation using the in-development fastai2 audio library and Google Cloud AI Platform notebooks.

The blog series is structured as follows, please follow the links for other sections:

1.  [Problem Definition, Analysis, Methods and Algorithms](https://mikful.github.io/blog/deep%20learning/fastai2/audio/markdown/2020/06/05/Udacity-Capstone-Part-1-Definition-and-Data.html)
2.  Methodology (Data Pre-processing, Implementation and Refinement)
3.  Results and Analysis

Links will be provided as the series progresses. Please see [the associated GitHub repository](https://github.com/mikful/udacity-mlend-capstone) for all notebooks.

A huge thanks goes to fastai and the fastai2 audio contributors for their amazing work.

<p>&nbsp;</p>




## IV. Methodology
### Data Pre-processing

**Fastai2 Audio**

The remarkable in-development fastai2 audio package was used to convert the audio files into mel-spectrogram 2D tensors on-the-fly, as a form of highly efficient data processing, rather than pre-processing and having to save spectrograms to a different dataset. This was done using the following process:

1. Create Pandas DataFrames for the files, suing the `train_curated.csv` and `train_noisy.csv` files provided by the competition, removing corrupted or empty files as given in the competition guidance:

```python
def create_train_curated_df(file, remove_files=[]):
    df_curated = pd.read_csv(file)
    df_curated['fname'] = '../data/train_curated/' + 	df_curated['fname'] 
    df_curated.set_index('fname', inplace=True)
    df_curated.loc[remove_files]
    df_curated.drop(index=remove_files, inplace=True)
    df_curated.reset_index(inplace=True)
    return df_curated

def create_train_noisy_df(file):
    df_noisy = pd.read_csv(file)
    df_noisy['fname'] = '../data/train_noisy/' + df_noisy['fname'] 
    return df_noisy


# Create Curated training set df
# Remove corrupt and empty files as per Kaggle guidance

remove_files = ['f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav']
remove_files = ['../data/train_curated/' + i for i in remove_files]
df_curated = create_train_curated_df('../data/train_curated.csv', remove_files=remove_files)
df_curated.head()
```

The DataFrames were then used to supply the fastai DataBlock API with the filenames, which could then be processed using the fastai2 audio `item_transformations` which are applied to each file before training. After significant testing iterations the audio transformation settings were chosen as follows:

```python
DBMelSpec = SpectrogramTransformer(mel=True, to_db=True) # convert to Mel-spectrograms

clip_length = 2 # clip subsection length in seconds
sr = 44100 # sample rate
f_min = 20 # mel-spectrogram minimum frequency
f_max = 20000 # mel-spectrogram minimum frequency
n_mels = 128 # mel-frequency bins, dictates the y-axis pixel size
hop_length = math.ceil((clip_length*sr)/n_mels)# determines width of image. for square to match n_mels, set math.ceil((clip_length*sr)/n_mels)
nfft = n_mels * 20 # = 2560 for higher resolution in y-axis
win_length = 1024 # sample windowing
top_db = 60 # highest noise level in relative db
```

* The `top_db` parameter setting of 60dB was important, as the noisy train set had high background noise (low signal-to-noise ratio) which with a higher setting lead to obscured features in the mel-spectrograms.

In addition to the mel-spectrogram settings above, the following additional item transformations were undertaken:

* `RemoveSilence`  - Splits the original signal at points of silence more than `2 * pad_ms`
* `CropSignal` - Crops a signal by `clip_length` seconds and adds zero-padding by default if the signal is less than `clip_length`
* `aud2spec` - The mel-spectrogram settings from above
* `MaskTime` - Uses Google's SpecAugment[^20] time masking procedure to zero-out time domain information as a form of data augmentation
* `MaskFreq` - Uses Google's SpecAugment[^20] frequency masking procedure to zero-out frequency domain information as a form of data augmentation

```python
item_tfms = [RemoveSilence(threshold=20),  
             CropSignal(clip_length*1000),  
             aud2spec,
             MaskTime(num_masks=1, size=8), MaskFreq(num_masks=1, size=8)]
```

**Batch Transforms**

In addition to the item transforms above, Batch Transforms were used as part of the DataBlock API, which are transformations applied per batch during training:

* `Normalize()` - normalizes the data taking a single batch's statistics
* `RatioResize(256)` - during training (other than the first 10 epochs of the noisy data for speed), the mel-spectrogram tensors were resized from 128x128px to 256x256px through bilinear interpolation as this has been shown to give gains in performance over simply creating a 256x256 tensor from the outset.
* `Brightness and Contrast` augmentations were also applied in the training cycles to improve performance
* 
```python
batch_tfms = [Normalize(),
              RatioResize(256),
              Brightness(max_lighting=0.2, p=0.75),
              Contrast(max_lighting=0.2, p=0.75)]
```

No further augmentations were applied as would otherwise be typical in many image classification processes. Many typical image augmentations applied  are not suitable for spectrogram representations of audio, for example, cropping/warping/rotating the spectrogram tensor would warp the relative frequency relationship and thus, would not gain any benefit at testing time. This was found to be true during tests of various transforms.

The above augmentations (prior to batch transformations), produced the following mel-spectrograms as 2D tensors (note the time and frequency masking augmentations):

<p>&nbsp;</p>
![Augmented Mel Spectrograms]({{ site.baseurl }}/images/udacity-capstone-series/aug-mel-spectrograms.jpg)
<p style="text-align: center;">Fig 8. Augmented Mel-spectrograms</p>
<p>&nbsp;</p>

[^20]: [Chan,Zhang,Chiu, Zoph,Cubuk,Le - 2019 - SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)



### Implementation

The data augmentations stated above were used to significantly improve the performance of the classifier during the following K-Fold training cycles.

The implemented training method was chosen based on the Competitions 6th place winner's technique[^20], however, only the first two stages were implemented as follows due to the cost requirements using GCP:

<p>&nbsp;</p>
![train-test-method]({{ site.baseurl }}/images/udacity-capstone-series/train-test-method.jpg)
<p style="text-align: center;">Fig 9. Train-Test-Prediction Stages</p>
<p>&nbsp;</p>

**Stage 1 - Noisy Training Set**

As can be seen below, the following 5-Fold training cycle was used on the noisy set. The indices of the DataFrame were shuffled to ensure the data splits were chosen at random, but without overlap using SKLearn's k-Folds module. The cycle began with 10 epochs of training at a higher learning rate and then 10 epochs of training at a lower learning rate (set after using fastai's learning rate finder during the testing stage) used to fine-tune the model's weights further. Please see the associated Jupyter Notebook for the training output.

The models were then saved for further training on the curated training set.

*Note: No MixUp augmentations were used on the Noisy Training set.*

```python
from sklearn.model_selection import KFold

# Declare Number Folds 
n_splits = 5

kf = KFold(n_splits=n_splits, random_state=42, shuffle=True) # random_state for repeatable results, shuffle indices

df = df_noisy # to use random subset, use  df = df_.sample(frac=0.5, replace=False, random_state=1) # take random subset of the noisy dataframe for faster training (otherwise need 6.5 hours for all folds with complete dataset)

for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
    print(f'\nNoisy Train Set - Fold {fold+1}/{n_splits}')

    def get_x(r): return r['fname']
    def get_y(r): return r['labels'].split(',') # split labels on ','
        
    def get_dls(train_cycle):
        if train_cycle == 1:
            batch_tfms = [Normalize(),
                          Brightness(max_lighting=0.2, p=0.75),
                          Contrast(max_lighting=0.2, p=0.75)]
            
        elif train_cycle == 2:
            batch_tfms = [Normalize(),
                          RatioResize(256), # progressive resize to 256x256px
                          Brightness(max_lighting=0.2, p=0.75),
                          Contrast(max_lighting=0.2, p=0.75)]
        
        dblock = DataBlock(blocks=(AudioBlock, MultiCategoryBlock),
                           splitter=IndexSplitter(valid_idx), # split using df index
                           get_x=get_x,
                           get_y=get_y,
                           item_tfms = item_tfms,
                           batch_tfms = batch_tfms
                          )
        return dblock.dataloaders(df, bs=64)
    
    dls = get_dls(train_cycle=1)

    dls.show_batch(max_n=6)

    model = xresnet50(pretrained=False, act_cls=Mish, sa=True, c_in=1, n_out=80) #create custom xresnet: 1 input channel,  80 output nodes, self-attention, Mish activation function
    model = convert_MP_to_blurMP(model, nn.MaxPool2d) # convert MaxPool2D layers to MaxBlurPool
    learn = Learner(dls, model=model, loss_func=BCEWithLogitsLossFlat(), opt_func = ranger, metrics=[lwlrap]) # pass custom model to Learner, no mixup for noisy set as fewer epochs

    learn.fit_flat_cos(10, lr=3e-3)
    
    print('Batch transforming images to 256x256px and training further.')
    
    dls = get_dls(train_cycle=2)
    learn.dls = dls
    learn.fit_flat_cos(10, lr=3e-3/3)

    print('Saving Learner...')
    learn.save(f'stage-1_noisy_fold-{fold+1}_sota2')
```



**Stage 2 - Curated Train Set**

After all 5 models had been trained on the Noisy set, the models were then trained on different 5-folds of the Curated Set. This essentially gave 5 distinct models, all trained on different data for later ensembling. 

*Note: MixUp data augmentations were applied to the Curated Train set, shown as training callback below. This is whereby two spectrogram tensors are combined into a single 2D tensor with a certain percentage blend (50% in this case), allowing the network to learn double the amount of features and labels per batch. This also provides a form of regularization for the model which improves generalization on the validation/test sets.*

```python
## K-Folds training loop

df = df_curated

for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
    print(f'\nCurated Train Set - Fold {fold+1}/{n_splits}')

    def get_x(r): return r['fname']
    def get_y(r): return r['labels'].split(',') # split labels on ','

    dblock = DataBlock(blocks=(AudioBlock, MultiCategoryBlock),
                       splitter=IndexSplitter(valid_idx), # split using df index
                       get_x=get_x,
                       get_y=get_y,
                       item_tfms = item_tfms,
                       batch_tfms = batch_tfms # including RatioResize(256)
                      )

    dls = dblock.dataloaders(df, bs=64)

    dls.show_batch(max_n=3)

    print(f'\nLoading Stage 1 model - fold {fold+1}.')
    
    model = xresnet50(pretrained=False, act_cls=Mish, sa=True, c_in=1, n_out=80) #create custom xresnet: 1 input channel,  80 output nodes, self-attention, Mish activation function
    model = convert_MP_to_blurMP(model, nn.MaxPool2d) # convert MaxPool2D layers to MaxBlurPool
    learn = Learner(dls, model=model, loss_func=BCEWithLogitsLossFlat(),  opt_func=ranger, metrics=[lwlrap]) # pass custom model to Learner, no mixup for noisy set as fewer epochs
    learn.load(f'stage-1_noisy_fold-{fold+1}_sota2')
    
    learn.dls = dls
    learn.add_cb(MixUp()) # add mixup callback
    
    print('\nTraining on Curated Set:')
    learn.fit_flat_cos(50, 3e-4)

    print('Saving model...')
    learn.save(f'stage-2_curated_fold-{fold+1}_sota2')
```



**Testing**

At the testing stage, Test-Time-Augmentations and ensembling the predictions of all 5 different Stage-2 models were used to improve the final predictions.

```python
# grab test filenames from submission csv
df_fnames = pd.read_csv('../data/sample_submission.csv')
fnames = df_fnames.fname
df_fnames = '../data/test/' + df_fnames.fname
print(df_fnames[:5])

# get predictions
for fold in range(n_splits):
    stage = 2
    print(f'Getting predictions from stage {stage} fold {fold+1} model.')

    learn = learn.load(f'stage-2_curated_fold-{fold+1}_sota2')

    dl = learn.dls.test_dl(df_fnames)
    
    # predict using tta    
    preds, targs = learn.tta(dl=dl)
    preds = preds.cpu().numpy()
    
    if fold == 0:
        predictions = preds
    else:
        predictions += preds


# Average predictions
predictions /= n_splits

# Create Submission DataFrame    
df_sub = pd.DataFrame(predictions)
df_sub.columns = learn.dls.vocab
df_sub.insert(0, "fname", 0)
df_sub.fname = fnames
df_sub.head()
```

The produced .csv file was then submitted to Kaggle.

[^21]: [https://github.com/ebouteillon/freesound-audio-tagging-2019](https://github.com/ebouteillon/freesound-audio-tagging-2019)



### Refinement

**Initial Model**

The  initial CNN architecture used was a pre-trained (on ImageNet) xresnet50 model for speed of iteration. This was trained on a single fold smaller subset of the Noisy data (ranging from 20-80% using the DataBlock API - `RandomSubsetSplitter()`function) used for faster iteration on the noisy subset, while all of the Curated data was used. The data augmentation settings were slightly different however, as non-square mel-spectrograms were used to see if larger spectrograms could give improved scores, which was the case, however, this was at the expense of training time.

This highest score achieved by any initial model, was an lwl-rap of 0.61013 on the test-set:


<p>&nbsp;</p>
![Initial Best Score]({{ site.baseurl }}/images/udacity-capstone-series/image-20200418112147418.png)
<p style="text-align: center;">Fig 10. Initial Best Score</p>
<p>&nbsp;</p>


This score, while not bad for a small amount of testing and still beating the competition baseline, was far from achieving near state-of-the-art performance. 

Test-Time-Augmentation was shown to provide a benefit of >3% improvement during further testing rounds using a single training fold on the noisy and curated datasets, shown as the top score in the following image:


<p>&nbsp;</p>
![Improvement using TTA]({{ site.baseurl }}/images/udacity-capstone-series/image-20200418113646639.png)
<p style="text-align: center;">Fig 11. Improvement using TTA</p>
<p>&nbsp;</p>


After reading further the writeups of the competition winners and high scorers [^19][^21][^22], it was decided that a K-Folds validation approach was required in order to substantially improve the performance. 

In addition, due to the large size of the spectrograms in the initial testing phase that would cause extremely slow training over so many folds (a 5x increase in epochs), these were replaced with smaller 128x128px (using 128 mel-bins and the settings shown in the above Data Pre-processing section). It was decided to first try training for a single fold on the Noisy set (90%/10% train/test split) and then split this model into 5 separate models for the further training on the Curated Set.

What's more, the pretrained xresnet50 model was replaced by the state-of-the-art xresnet50 model described in the previous Section II: Algorithms and Techniques. This was in line with the allowance of only non-pretrained models in the competition and was also shown to provide small improvements over the pretrained xresnet50 over long enough training cycles, such that the non-trained units could effectively learn, as shown below:


<p>&nbsp;</p>
![Improvement with SOTA model]({{ site.baseurl }}/images/udacity-capstone-series/image-20200418113528769.png)
<p style="text-align: center;">Fig 12. Improvement using SOTA model</p>
<p>&nbsp;</p>

Finally, a full 5-Fold Cross-Validation training was undertaken for both the Noisy and Curated set as detailed in Figure 9 in the Implementation section above, with some tweaks to the spectrograms settings, i.e. using `top_dB` of 60 to ensure only the most prominent Noisy Set features were captured by the mel-spectrograms. This this approach achieved the final score of 0.69788, a marked improvement that would have gained a bronze-medal position in the competition and could certainly be improved upon further.

[^19]: [https://github.com/lRomul/argus-freesound](https://github.com/lRomul/argus-freesound)
[^22]: [https://medium.com/@mnpinto/multi-label-audio-classification-7th-place-public-lb-solution-for-freesound-audio-tagging-2019-a7ccc0e0a02f](https://medium.com/@mnpinto/multi-label-audio-classification-7th-place-public-lb-solution-for-freesound-audio-tagging-2019-a7ccc0e0a02f)



## Up Next

In Part 3 of this blog series, we'll look at the final results and how these could be improved upon.

If you have any questions or feedback about this post, I'd be very happy to hear them. Please contact me at my GitHub or Twitter using the links below.




## References: