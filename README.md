# bambas
Code for SemEval Task4 Subtask 1

## Environment

* [python == 3.8](https://www.python.org/downloads/release/python-3818/)
* [pip3](https://pip.pypa.io/en/stable/cli/pip_install/pipe)
* [pipenv](https://pypi.org/project/pipenv/) or [conda](https://docs.conda.io/projects/miniconda/en/latest/) (optional but **strongly** recommended) - for environment isolationcon

## Installation

First, clone `sklearn-hierarchical-classification` repository:

```sh
$ git clone https://github.com/lfmatosm/sklearn-hierarchical-classification
```

### With `pip`
```sh
$ pip install -r requirements.txt
```

### With `pipenv`
```sh
$ pipenv shell
$ pipenv install
```

If you encounter any problems related to installing `sklearn-hierarchical-classification` with `pipenv`, just ignore it.

After the previous steps, use `pip` to install the local repository:
```sh
$ pip install ../sklearn-hierarchical-classification # point to the cloned repository path
```

## Running
For a working Google Colab example, please refer to [this notebook](./Fine_tuning_+_feature_extraction_+_class.ipynb).

### Fine-tuning
```sh
$ python -m src.fine_tuning \
  --model xlm-roberta-base \
  --dataset ptc2019 \
  --fine_tuned_name xlm-roberta-base-ptc2019 \
  --save_model
```

### Feature-extraction
```sh
$ python -m src.feature_extraction \
  --model xlm-roberta-base \
  --dataset semeval2024 \
  --extraction_method cls
```

Or if you want to use specific hidden-layers:
```sh
$ python -m src.feature_extraction \
  --model xlm-roberta-base \
  --dataset semeval2024 \
  --extraction_method layers \
  --layers 4 5 6 7 \
  --agg_method "avg"
```

### Classification
```sh
$ python -m src.classification \
  --classifier "HiMLP" \
  --dataset semeval2024 \
  --train_features "./feature_extraction/train_features.json" \
  --test_features "./feature_extraction/test_features.json" \
  --dev_features "./feature_extraction/dev_features.json" \
  --seed 1
```