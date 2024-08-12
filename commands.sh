#!/bin/bash

# ft extraction using semeval base
python -m src.feature_extraction \
 --dataset semeval2015_paraphrased2 \
 --model jhu-clsp/bernice \
 --extraction_method cls \
 --output_dir semeval2015_paraphrased2

# semeval 2016
python -m src.classification \
 --dataset semeval2015_paraphrased2 \
 --train_features feature_extraction/semeval2015_paraphrased2/train_features.json \
 --test_features feature_extraction/semeval2015_paraphrased2/test_features.json \
 --dev_features feature_extraction/semeval2015_paraphrased2/dev_features.json \
 --classifier LogisticRegression --binary --oversampling None

python -m src.classification \
 --dataset semeval2015_2 \
 --train_features feature_extraction/semeval2015_2/train_features.json \
 --test_features feature_extraction/semeval2015_2/test_features.json \
 --dev_features feature_extraction/semeval2015_2/dev_features.json \
 --classifier LogisticRegression --binary --oversampling SMOTE --sampling_strategy 0.3455