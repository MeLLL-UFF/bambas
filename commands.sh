#!/bin/bash

# ft extraction using semeval base
python -m src.feature_extraction \
 --dataset semeval2015_2 \
 --model jhu-clsp/bernice \
 --extraction_method cls \
 --output_dir semeval2015_2

# run classification with ros
python -m src.classification \
 --dataset semeval2015_2 \
 --train_features feature_extraction/semeval2015_2/train_features.json \
 --test_features feature_extraction/semeval2015_2/test_features.json \
 --dev_features feature_extraction/semeval2015_2/dev_features.json \
 --classifier LogisticRegression --binary --oversampling RandomOverSampler --sampling_strategy 1

# run classification with smote
python -m src.classification \
 --dataset semeval2015_2 \
 --train_features feature_extraction/semeval2015_2/train_features.json \
 --test_features feature_extraction/semeval2015_2/test_features.json \
 --dev_features feature_extraction/semeval2015_2/dev_features.json \
 --classifier LogisticRegression --binary --oversampling SMOTE --sampling_strategy 1

# run classification on paraphrased dataset
python -m src.classification \
 --dataset semeval2015_paraphrased2_1to4_selected \
 --train_features feature_extraction/semeval2015_paraphrased2_1to4_selected/train_features.json \
 --test_features feature_extraction/semeval2015_paraphrased2_1to4_selected/test_features.json \
 --dev_features feature_extraction/semeval2015_paraphrased2_1to4_selected/dev_features.json \
 --classifier LogisticRegression --binary --oversampling None