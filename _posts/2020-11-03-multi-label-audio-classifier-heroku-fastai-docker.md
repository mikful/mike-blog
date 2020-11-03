---
toc: false
layout: post
categories: [deep learning, fastai2, audio, heroku, docker markdown]
title: An Audio Classification Web-app using Heroku, Docker and fastai

---

This is a short blog post about how to create a multi-label audio classification app using fastai and the Heroku platform, which is available to try out here:

https://audio-recorder-ctmf.herokuapp.com/

In this blog we'll quickly run through how to setup a Docker container environment on Heroku, and the Python server application that will perform the inference.

The full code is available at the following github repo: https://github.com/mikful/audio-app-mf-ct-heroku. Please note this was made in collaboration with https://github.com/cltweedie creating the front-end web-app.

# Model Training

Naturally, we'll need a trained model to perform our inference. This blog isn't dedicated to the training procedure, rather the deployment, however, to see the training procedure please view the notebooks within the nbs folder of the repo. 

Some basic information about the training and model:

* The multi-label [Freesound 2019 Kaggle Competition](https://www.kaggle.com/c/freesound-audio-tagging-2019) dataset (training and curated test set) was used to the train the model. The audio data is labelled using a vocabulary of 80 labels from Google’s [AudioSet Ontology](https://research.google.com/audioset////////ontology/index.html), giving a wide variety of audio classes for a general-classifier.
* The [fastaudio](https://fastaudio.github.io/) library was used with mel-spectrograms being fed into a 2D xresnet18 architecture
* The model was trained for 80 epochs from scratch, with noise and volume augmentations and time and frequency masking augmentations
* Mixup augmentations were also applied
* A cyclical learning rate schedule was used
* A label-weighted label-ranking average precision (lwl-rap) metric was used to delineate the best achieving model (giving an lwl-rap of 75.93%, and over 99% multi-label accuracy)

# Heroku

Deploying to Heroku was straightforward, despite the fact that the fastaudio library at did not have a pypi installation package. As such, the standard method of simply creating a Python app from the requirements.txt file in the root directory (which is normally detected upon connecting the github repo adn automatically created) was not possible. To work around this, a `heroku.yml` file was created, such that a Docker environment could be built initially from the `requirements.txt` file and then install the fastaudio package separately afterwards.

## Deployment steps

1. Create the heroku app and follow the steps in /deploy/heroku-git i.e.:

- `$ heroku login`
- `$ cd my-project/`
- `$ heroku git:remote -a your-heroku-app-name`

1. Now follow the details regarding [Building Docker Images with heroku.yml](https://devcenter.heroku.com/articles/build-docker-images-heroku-yml#getting-started) once you have the `heroku.yml` ready. i.e. Set the stack of your app to container: `heroku stack:set container`
2. Push your app to Heroku: `git push heroku master`
3. Connect your github repo directly for push deployments

## Dockerfile

We can see from the [dockerfile](https://github.com/mikful/audio-app-mf-ct-heroku/blob/master/Dockerfile) the install of the forked fastaudio library in the line:

```python
RUN pip install git+https://github.com/mikful/fastaudio.git
```

## Inference

The inference code can be seen within [server.py](https://github.com/mikful/audio-app-mf-ct-heroku/blob/master/app/server.py) file. This was based on the code used within my [Orchid Classifier App with fastai, Render and Flutter](https://mikful.github.io/blog/fastai/jupyter/render/flutter/2020/09/16/orchid-classifier-fastai-render-flutter-2.html), but  has been modified to take an audio file stream and perform inference on the bytes data.

The main functions within the `server.py` file are fairly self evident as per the Orchid Classifier app, in that it downloads the `export.pkl` file and then loads it into a new fastai `Learner`.

Now we define the location of our exported fastai Learner. We first upload this to a Google Cloud Storage bucket - we need to ensure the permissions on the file are set such that we can download this, which can be done in the file settings in the bucket (for more details [see here](https://cloud.google.com/storage/docs/access-control/making-data-public)).

Then we set the URL within the script:

```python
path = Path(__file__).parent
export_file_url = 'https://storage.googleapis.com/example-bucket/export.pkl' # google cloud bucket / file url
export_file_name = 'export.pkl'
```

For the inference, the main function is contained within the following code section that takes the audio file bytes from the post request and performs the inference on it. However, this time the predictions are multi-label outputs, so we will create a tuple of tuples containing the predictions and the class label for displaying.  The prediction is then returned in the JSON response.

```python
@app.post("/analyze")
async def analyze(file: bytes = File(...)):
    wav = BytesIO(file)
    utc_time = str(int(time.time()))
    sound_file = "tmp/sound_" + utc_time + ".wav"
    with open(sound_file, mode='bx') as f: f.write(wav.getvalue())
    prediction, idx, preds =  learn.predict(Path(sound_file))
    predictions_ordered = learn.dls.vocab[np.argsort(preds.squeeze()).squeeze()][::-1] # descending order
    conf_sorted = np.sort(preds.squeeze()).squeeze()[::-1] # descending order
    results_ordered = tuple(zip(predictions_ordered, np.rint(conf_sorted*100).tolist()))
    return JSONResponse({'classifications': json.dumps(results_ordered)})
```

In addition, we need to set out ports correctly:

```python
Port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=Port, log_level="info") #heroku
```

And that's it - we have a working audio classifier webapp, with very high accuracy!

This was a fun little project which had a few challenges in terms of the deployment of a non pypi library. I hope to expand it with some real-time inference at a later date.

Please feel to use the webapp and I'd love to hear any feedback you have, so contact me at one of the links given below.