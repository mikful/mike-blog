---
toc: false
layout: post
description: A Udacity Machine Learning Engineer Nanodegree Capstone Project
categories: [deep learning, fastai2, audio, markdown]
title: Multi-Label Auto-Tagging of Noisy Audio Using fastai2 - Part 1 - Definition and Data
---



Welcome to Part 1 of a blog series based on my Udacity Machine Learning Engineer Nanodegree Capstone project.  This initial section deals with the problem definition, outlines the solution approach using the in-development fastai2 audio library and discusses the dataset.

The blog series will be structured as follows:

1.  Problem Definition, Analysis and Methods and Algorithms
2. [Methodology (Preprocessing, Implementation and Refinement)](https://mikful.github.io/blog/deep%20learning/fastai2/audio/markdown/2020/06/12/Udacity-Capstone-Part-2-Implementation.html)
3. Results and Analysis

Links will be provided as the series progresses. Please see [the associated GitHub repository](https://github.com/mikful/udacity-mlend-capstone) for all notebooks.

A huge thanks goes to fastai and the fastai2 audio contributors for their amazing work.

<p>&nbsp;</p>


## I. Problem Definition
### Overview

The sub-field of Machine Learning known as Machine Listening is a burgeoning area of research using signal processing for the automatic extraction of information from sound by a computational analysis of audio. There are many different areas of research within this field as demonstrated by the latest Detection and Classification of Acoustic Scenes and Events (DCASE) 2020 Challenge[^1], a machine learning challenge dedicated to the research and development of new methods and algorithms for audio. These include:

* Acoustic Scene Classification
* Sound Event Detection and Localization
* Sound Event Detection and Separation in Domestic Environments
* Urban Sound tagging
* Automated Audio Captioning

As an acoustic engineer, I am extremely intrigued by this new field. Recent developments in machine learning algorithms have allowed significant progress to be made within this area, with the potential applications of the technology being wide and varied and meaning the tools could prove to be extremely useful for the acoustic practitioner amongst many other uses.

The in-development user-contributed fast.ai2 Audio library[^2] inspired me to undertake the development of a deep learning audio-tagging system for this Udacity Capstone project, described herein.

[^1]: [http://dcase.community/challenge2020/index](http://dcase.community/challenge2020/index)
[^2]: [https://github.com/rbracco/fastai2_audio](https://github.com/rbracco/fastai2_audio)



### Problem Statement

The Freesound Audio Tagging 2019 Kaggle Competition provides the basis for my research project[^3]. 

The challenge is to develop a system for the automatic classification of multi-labelled audio files within 80 categories, that could potentially be used for automatic audio/video file tagging with noisy or untagged audio data. This has historically been investigated in a variety of ways:

* Conversion of audio to mel-spectrogram images fed into CNNs
* End-to-End Deep learning
* Custom architectures involving auto-encoders
* Features representation transfer learning with custom architectures and Google's Audioset

In addition, the classification of weakly labelled data from large-scale crowdsourced datasets provides a further problem for investigation[^4]. The problem is clearly quantifiable in that a number of accuracy metrics could be used to quantify the accuracy of the model's predictions, described below.

The competition dataset comprises audio clips from the following existing datasets:

- "Curated Train Set" - Freesound Dataset ([FSD](https://annotator.freesound.org/fsd/)): a smaller dataset collected at the [MTG-UPF](https://www.upf.edu/web/mtg) based on [Freesound](https://freesound.org/) content organized with the [AudioSet Ontology](https://research.google.com/audioset////////ontology/index.html) and manually labelled by humans.  4964 files.
- "Noisy Train Set" - The soundtracks of a pool of Flickr videos taken from the [Yahoo Flickr Creative Commons 100M dataset (YFCC)](http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/) which are automatically labelled using metadata from the original Flickr clips. These items therefore have significantly more label noise than the Freesound Dataset items. 19815 files.

The data comprises 80 categories labelled according to Google's Audioset Ontology [^3] with ground truth labels provided at the clip level. The clips range in duration between 0.3 to 30s in uncompressed PCM 16 bit, 44.1kHz mono audio files.

[^3]: [https://www.kaggle.com/c/freesound-audio-tagging-2019/overview](https://www.kaggle.com/c/freesound-audio-tagging-2019/overview) 
[^4]: [Learning Sound Event Classifiers from Web Audio with Noisy Labels - Fonseca et al. 2019](https://arxiv.org/abs/1901.01189)



### Solution Statement

With the above competition requirements in mind, the proposed solution was followed and was undertaken initially within a Pytorch AWS SageMaker notebook instance using Jupyter Notebooks, and further, using a Google Cloud Platform AI Notebook using fastai2 and fastai2 audio libraries , due to the extra credits required for the long training times on a GPU instance:

1. The data will be downloaded from Kaggle into the chosen platform AWS SageMaker / GCP AI Notebook instance
2. Exploratory Data Analysis - The dataset will be downloaded such that the file metadata can be extracted, in order to confirm:  sample rates, bit-rates, durations, channels (mono/stereo) for each file in order to direct initial signal processing stage and approach towards the dataset splitting. In addition, the statistics of the file labels will be analysed.
3. Model Development:
   * The fastai2 and fastai2 audio libraries will be installed
   * The fastai2 audio library will be used for the data processing, in order to convert the audio files into tensor representations of mel-spectrograms on-the-fly, rather than in a separate pre-processing stage. This is a significant benefit of the library in terms of allowing quick experimentation and iteration within the model development over other methods such as converting all audio files to mel-spectrogram images separately.
   * In-line with the competition rubric, a non-pretrained convolutional neural network (CNN) using the fastai2 library for PyTorch will be developed using state-of-the-art methods and additions.
   * The model will be trained on the "Noisy Train" set in a 5-Folds Cross Validation manner, using Sci-Kit Learn's K-Fold model selection module[^5].
   * The results of these 5 models will be used to train on the "Curated Train" set in the same 5-Folds Cross Validation manner as the Curated Train set in order to gain 5 separate models.
   * Test-Time-Augmentation (TTA) will be used to gain averaged predictions from all 5 final models on the test set. The predictions will be submitted as a Late-Submission for the analysis of the results.
   * This will be repeated, with tweaks to the model augmentations in order to try to improve the results iteratively.

[^5]: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)



### Metrics

Due to the advancement of multi-label audio classification in recent years, a simple multi-label accuracy metric was not used within the Kaggle competition, as performances of the systems can easily exceed 95% within a few epochs of training.

As such, the competition used label-weighted label-ranking average precision (a.k.a lwl-rap) as the evaluation metric. The basis for the metric, the label-ranking average precision algorithm, is described in detail within the Sci-Kit Learn implementation[^6]. The additional adaptations of the metric are to provide the average precision of predicting a ranked list of relevant labels per audio file, which is a significantly more complex problem to solve than a standard multi-label accuracy metric. The overall score is the average over all the labels in the test set, with each label having equal weight (rather than equal weight per test item), as indicated by the "label-weighted" prefix. This is defined as follows[^7]: 
<p>&nbsp;</p>

$$
lwlrap = \frac{1}{\sum_{s} \left | C(s) \right |}\sum_{a}\sum_{e\epsilon\ C(s)}Prec(s,c)
$$

<p>&nbsp;</p>
where $$Prec(s,c)$$ is the label-ranking precision for the list of labels up to class $$c$$ and the set of ground-truth classes for sample $$s$$ is $$C(s)$$. $$\mid C(s)\mid$$ is the number of true class labels for sample $$s$$. 

The Kaggle competition provides a Google Colab example implementation[^8]. 

[^6]: [https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision](https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision) 
[^7]: [Fonseca et al. - *Audio tagging with noisy labels and minimal supervision*. In Proceedings of DCASE2019 Workshop, NYC, US (2019).](https://arxiv.org/abs/1906.02975) 
[^8]: [https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8](https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8)




## II. Analysis
### Data Exploration

The datasets were downloaded from Kaggle using the Kaggle API and analysed within a Jupyter Notebook. 

The first stage of the process was to understand the dataset more fully. Fortunately, due to being a Kaggle Competition dataset it was well documented and clean in terms of organization.

Downloading the dataset was undertaken using guidance given within the Kaggle Forums[^9] directly into the SageMaker/GCP Instance storage for easy access.

The files were then unzipped for the EDA. For further details, please see the notebook directly.

**Pandas and Pandas Profiling**

In order to undertake the analysis of the data, the numerical data analysis packages Pandas and Pandas Profiling were used.

Pandas Profiling[^10] is an extremely useful add-on package to Pandas, which creates HTML profile reports directly from Pandas DataFrames quickly and easily. From the provided .csv files file category labels were analysed and, in addition, the audio file meta-data was extracted (i.e. sample rates, bit-rates, durations, number of channels). 

<p>&nbsp;</p>
![Pandas Profiling metadata]({{ site.baseurl }}/images/udacity-capstone-series/image-20200417150251804.png)
<p style="text-align: center;">Fig 1. An example Pandas DataFrame of extracted audio file info</p>
<p>&nbsp;</p>

Using these two packages the following was found.

**Curated Train Data**

For the Curated Train dataset, it was found that the bit-rate was a constant 16bits, the channels a constant 1 (mono), constant sample rate of 44.1kHz and that there were 213 different tagging combinations of the 80 audio labels over the total file count (4964 files):  

<p>&nbsp;</p>
![Pandas Profiling - Curated set]({{ site.baseurl }}/images/udacity-capstone-series/image-20200417150839322.png)
<p style="text-align: center;">Fig 2. Pandas Profiling for the Curated Train Data set</p>
<p>&nbsp;</p>

In terms of the file durations, the average file length was 7.63 seconds and the files ranged between just over 0 and 30 seconds long, with the lengths predominantly in the 0-5 seconds length range. This will affect the mel-spectrogram processing of the data, i.e. we will need to ensure a sufficient amount of both the longer and smaller audio files are taken, in order for the feature learning of the CNN to be accurate. 

<p>&nbsp;</p>
![Pandas Profiling - Curated set]({{ site.baseurl }}/images/udacity-capstone-series/image-20200417151105376.png)
<p style="text-align: center;">Fig 3. Pandas Profiling information for the audio file durations</p>
<p>&nbsp;</p>

**Noisy Train Data**

As with the Curated dataset, with the Noisy Train dataset it was found that the bit-rate was a constant 16bits, the channels a constant 1 (mono), constant sample rate of 44.1kHz. However, in this dataset there were 1168 different tagging combinations of the 80 audio labels over the total file count (19815 files): 

<p>&nbsp;</p>
![Pandas Profiling - Noisy set]({{ site.baseurl }}/images/udacity-capstone-series/image-20200417151545657.png)
<p style="text-align: center;">Fig 4. Pandas Profiling for the Noisy Train dataset</p>
<p>&nbsp;</p>

The Noisy Train dataset average file length was significantly longer on average than the Curated set at 14.6s long, however, the files ranged between 1 and 16 seconds long. There is therefore a significant difference in terms of length between the two datasets. 

<p>&nbsp;</p>
![Pandas Profiling - Durations]({{ site.baseurl }}/images/udacity-capstone-series/image-20200417151832940.png)
<p style="text-align: center;">Fig 5. Pandas Profiling information for the audio file durations</p>
<p>&nbsp;</p>

In addition, as the name implies, the Noisy Train set files have a significantly higher noise floor than the Curated Train set due to the provenance of the files.


### Data Visualisation

The following figure clearly illustrates the differences between the difference in durations of audio files between the two datasets: 

<p>&nbsp;</p>
![Pandas Profiling - Durations]({{ site.baseurl }}/images/udacity-capstone-series/dataset-length-comp.jpg)
<p style="text-align: center;">Fig 6. Train vs Noisy dataset durations (x-axis = seconds)</p>
<p>&nbsp;</p>

Therefore, in the development of the model the following factors will need to be considered:

* Noise floor differences between the curated and noisy train set will affect how the signals are clipped to shorter lengths to feed into the CNN.
* The average lengths also have a high range of values both over the the individual datasets and between the curated and noisy set, we will need to ensure the main recorded features corresponding to the file labels of each recording are kept within any clipped sections to produce the mel-spectrograms.



## III. Algorithms and Techniques

**Mel-Spectrograms**

This signal processing stage will involve trimming (to ensure uniform duration) in order be converted to uniform length log-mel-spectrogram representations of the audio. A log-mel-spectrogram is a spectrogram representation of the audio i.e. a frequency-domain representation based on the Fourier Transform (x-axis = time, y axis = frequency and colour depth/pixel value = relative sound intensity) which has been converted to the Mel scale on the y-axis by a non-linear transform in order to be more representative of the highly non-linear magnitude and frequency sensitivities of the human hearing[^11]. The chosen settings will be discussed and shown further in the Data Preprocessing section. 

<p>&nbsp;</p>
![Waveform to Mel spectrogram]({{ site.baseurl }}/images/udacity-capstone-series/wav-melspec-conversion.jpg)
<p style="text-align: center;">Fig 7. Conversion from Waveform to Mel-spectrogram representation</p>
<p>&nbsp;</p>


**Convolutional Neural Network (CNN)**

The length uniformity of the audio clips in is important, as it allows Rank-2 tensors of the mel-spectrograms to be fed in batches into the CNN. The model variety used was as follows, based on the state of the art findings of the fastai community[^12] and other research described below. The model and architecture used the following settings:

* Architecture: fastai2's XResNet50 based on the Bag of Tricks[^13] research paper which includes tweaks to the optimisation methods for higher performance. ResNets use skip connections in order to allow propagation of information more deeply into the architecture, giving significant speed improvements for deeper networks while allowing the gradient descent to backpropagate through the network efficiently which aids in increasing training accuracy. This has further been augmented in the Bag of Tricks paper, whereby the residual block convolutional layers have been re-arranged such that further efficiency gains are made.

- Activation Function: Mish[^14] which has been shown to provide performance improvements over the standard ReLU activation function due to its smoothing of the activations rather than the cut-off of the ReLU function for values below 0.

- Optimizer Function: Ranger which is a combination of the RAdam[^15] and Lookahead[^16] optimizer functions. These functions work as a searching pair, whereby one learner goes ahead of the other to explore the function topography, such that traps involving local minima can be avoided.

- Layer tweaks: Self-Attention Layers[^17] which allow the network to focus on a subset on the features learned to be most pertinent to the data label.

- Replacing Max Pooling Layers with "MaxBlurPool" layers for better generalization

- Flat-Cosine decay learning rate scheduling

  

**K-Folds Validation**

Sci-Kit Learn's KFolds Validation function was used to split the datasets into 5 folds, to allow all of the available data to be used in the training and to further allow the 5 created models to give ensembled predictions on the Test set, which provides a significant performance improvement over a single model.

**MixUp**

MixUp, whereby two spectrograms are combined to form a third, was also used during the longer Curated Training Set procedure. Detailed further below.

**Test-Time Augmentation (TTA)**

In addition to the methods outlined above, Test-Time augmentations were applied to the test set, such that the data transformations were used as part of the testing procedure in order to give a further performance boost.

[^9]: [https://www.kaggle.com/c/deepfake-detection-challenge/discussion/129521](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/129521)
[^10]: [https://github.com/pandas-profiling/pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)
[^11]: Computational Analysis of Sound Scenes and Events, pg. 22 - Virtanen et al.
[^12]:  [https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/blob/master/Computer%20Vision/04_ImageWoof.ipynb](https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/blob/master/Computer Vision/04_ImageWoof.ipynb)
[^13]: [He, Tong, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. 2018. “Bag of Tricks for Image Classification with Convolutional Neural Networks.” *CoRR* abs/1812.01187](http://arxiv.org/abs/1812.01187)
[^14]: [Misra, Diganta. 2019. “Mish: A Self Regularized Non-Monotonic Neural Activation Function.”](http://arxiv.org/abs/1908.08681)
[^15]: [Liyuan Liu et al. 2019 - On the Variance of the Adaptive Learning rate and Beyond](https://arxiv.org/abs/1908.03265)
[^16]: [Zhang, Lucas Hinton, Ba - Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)
[^17]: [Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena 2018 - Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
[^18]: [Zhang R - 2019- Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/pdf/1904.11486) 



### Benchmark

The Baseline performance for the Kaggle Competition was set at 0.53792 which provided a minimum target. The winner[^19] of the competition achieved 0.75980, which provided the upper target. The details of the winning model and training method can be found on the linked GitHub page, but for brevity, the basic details of the system from the GitHub repo, were as follows:

> - Log-scaled mel-spectrograms
> - CNN model with attention, skip connections and auxiliary classifiers
> - SpecAugment, Mixup augmentations
> - Hand relabeling of the curated dataset samples with a low score
> - Ensembling with an MLP second-level model and a geometric mean blending

[^19]: [https://github.com/lRomul/argus-freesound](https://github.com/lRomul/argus-freesound)



## Up Next

In [Part 2](https://mikful.github.io/blog/deep%20learning/fastai2/audio/markdown/2020/06/12/Udacity-Capstone-Part-2-Implementation.html) of this blog series, we will look at the methodology and implementation of training the model and improving it iteratively.



If you have any questions or feedback about this post, I'd be very happy to hear them. Please contact me at my GitHub or Twitter using the links below.




## References:


