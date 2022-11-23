# GPT-2 German Version Fine-tuned on News Articles

## Idea
The following project was part of my master's thesis on Natural Language Generation in German. 
The goal was to fine-tune an existing GPT-2 model in German and test the results to see how the model would performe.
The base model has been released on Github and is part of the Transformers module. 
It can be found [here](https://github.com/stefan-it/german-gpt2).
```
@software{stefan_schweter_2020_4275046,
  author       = {Stefan Schweter},
  title        = {German GPT-2 model},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.4275046},
  url          = {https://doi.org/10.5281/zenodo.4275046}
}
```

## Installation
For the Code to work you need to install the following modules:
- `Python 3.9`
- `Transformers`
- `Datasets`


## Training Corpora
The corpora used for the fine-tuning of the model is part of the MLSum Dataset and contains over 20'000 news-articles in German,
which were released in the Süddeutsche Zeitung. It is part of the Dataset module from Huggingface, which makes it easy to use.
The dataset card and additional information can be found [here](https://huggingface.co/datasets/mlsum).


## Usage
To train the model run news_gpt.py. If you wish to make changes to the training, you can alter the training parameters.
They were chosen to fit the server I used for training, if you can use stronger GPU's or even TPU's, it's possible to train for example with bigger batches.
The weights will be stored in a separate folder, called `/gpt2-news`.

After training new articles can be generated using article.py
At the moment the code takes the beginning of an article which is part of the MLSum dataset. If you want the model to generate something else, you can just change that.
You're also free to change the maximum length, but keep in mind, that the number is referring to the number of tokens and not to the number of words.

