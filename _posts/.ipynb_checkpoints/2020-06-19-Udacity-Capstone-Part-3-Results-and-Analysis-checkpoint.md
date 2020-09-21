---
toc: false
layout: post
description: A Udacity Machine Learning Engineer Nanodegree Capstone Project
categories: [deep learning, fastai2, audio, markdown]
image: ../images/udacity-capstone-series/melspec5-part3.png
title: Multi-Label Auto-Tagging of Noisy Audio Using fastai2 - Part 3 - Results and Analysis
---
Welcome to part 3 of a blog series based on my Udacity Machine Learning Engineer Nanodegree Capstone project.  This section defines the model implementation using the in-development fastai2 audio library and Google Cloud AI Platform notebooks.

The blog series is structured as follows, please follow the links for other sections:

1.  [Problem Definition, Proposed Solution and Data Exploration](https://mikful.github.io/blog/deep%20learning/fastai2/audio/markdown/2020/06/05/Udacity-Capstone-Part-1-Definition-and-Data.html)
2. [Methodology and Implementation](https://mikful.github.io/blog/deep%20learning/fastai2/audio/markdown/2020/06/12/Udacity-Capstone-Part-2-Implementation.html)
3.  Results and Analysis

Links will be provided as the series progresses. Please see [the associated GitHub repository](https://github.com/mikful/udacity-mlend-capstone) for all notebooks.

A huge thanks goes to fastai and the fastai2 audio contributors for their amazing work.


## IV. Results

### Model Evaluation and Validation

The procedures outlined in the above sections were used to obtain a final prediction score of 0.69788.

![Final Score]({{ site.baseurl }}/images/udacity-capstone-series/image-20200418113906272.png)

​																	Fig 13. Final Prediction Score, lwl-rap



### Justification

The Competition baseline score of 0.53792 was beaten by a considerable margin of 16%. The  winning score of 0.75980 was not achieved, with a shortfall of 6%, however, the winning model and the top-scoring models documented [^19] [^21][^22] used training stages of 3x to 5x as long as the one implemented herein.

Additionally, the amount of pre-processing within the solution presented was negligible, afforded by the very impressive fastai2 audio library, whereas other solutions involved the significant extra time and storage usage of conversion from audio files to spectrogram images

As such, it is considered that the model performance is more than satisfactory for the task of auto-tagging audio files and further testing would be needed to see if it could achieve the same performance as the top-scorers.




## V. Conclusion

### Free-Form Visualization

One very clear visual element of the datasets is the difference in the noise level and quality of the recordings between the two datasets. This is shown clearly below:

![noisy-curated-comp]({{ site.baseurl }}/images/udacity-capstone-series/noisy-curated-comp.jpg)

​											Fig 14. Differences in noise level between Curated and Noisy Set



The impact of this was that, when using the noisy dataset for training alone, the lwl-rap score was approximately 20% lower than using only the curated dataset for training. As such, the two stage training method was used for bigger performance gains.



### Reflection

The stages involved in the production of this model were varied, each step requiring careful consideration of the data and audio file to spectrogram augmentations.

The most interesting aspects of the project were considered to be the time/performance cost balance required in the data augmentations' effect on the training. It was extremely interesting to also see the stark impact of the K-Folds Validation training procedure and Test-Time-Augmentations on the final score.

The main difficulties in the project were the balancing of trying to achieve a good score while minimising training and iteration costs on the Google Cloud Platform. Given the nature of the problem, the training times were necessarily long and this was not anticipated fully at the outset of choosing the problem set. In a production setting this would be a significant consideration to bear in mind.

The final model is considered to be of good performance and will serve to inform future model development for production-stage inference of audio auto-tagging. Naturally, the TTA and ensembling of predictions means this model should only be used in an offline scenario for complex and noisy audio tagging, however, it should be noted that using this system with a standard multi-label accuracy metric, a score of over 95% was achieved within the first few epochs of training on both the Curated and Noisy Train sets.

As such, a similar approach, perhaps using a different metric (standard multi-label accuracy), could be used for a deployed inference model to be used within applications for audio classification e.g. for bird sounds, which is an area of interest of the author. 

### Improvement

It is considered the final prediction score could be improved in a number of ways:

* Longer training times
* Further fine-tuning of the mel-spectrogram settings
* further ensembling of models
* further data engineering, such as further training on correctly predicted noisy data [^21]

The majority of the techniques were implemented within the development of the model, however, in the best performing models in the competition, more advanced data engineering was undertaken such as cross-referencing prediction scores between models and only doing further training on the audio files that were correctly selected over a certain threshold across models. This would require significant extra development that the author did not achieve within the time frame available.

## Conclusion

That concludes the 3 part series on my Udacity Machine Learning Engineer Nanodegree Capstone project. It was a fantastic learning experience and I hope one that is also useful for others.

If you have any questions or feedback about this post, I'd be very happy to hear them. Please contact me at my GitHub or Twitter using the links below.



## References:

[^19]: [https://github.com/lRomul/argus-freesound](https://github.com/lRomul/argus-freesound)

[^21]: [https://github.com/ebouteillon/freesound-audio-tagging-2019](https://github.com/ebouteillon/freesound-audio-tagging-2019)

[^22]: [https://medium.com/@mnpinto/multi-label-audio-classification-7th-place-public-lb-solution-for-freesound-audio-tagging-2019-a7ccc0e0a02f](https://medium.com/@mnpinto/multi-label-audio-classification-7th-place-public-lb-solution-for-freesound-audio-tagging-2019-a7ccc0e0a02f)