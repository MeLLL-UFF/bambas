#!/bin/bash

# semeval
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/semeval2024/bernice/1707079469_jhu-clsp-bernice_train_features.json \
 --test_features feature_extraction/semeval2024/bernice/1707079469_jhu-clsp-bernice_test_features.json \
 --dev_features feature_extraction/semeval2024/bernice/1707079469_jhu-clsp-bernice_dev_features.json \
 --classifier LogisticRegression

# semeval_internal trained with train test dev
python -m src.classification \
 --dataset semeval_internal \
 --train_features feature_extraction/semeval_internal/bernice/1707069752_jhu-clsp-bernice_train_features.json \
 --test_features feature_extraction/semeval_internal/bernice/1707069752_jhu-clsp-bernice_test_features.json \
 --dev_features feature_extraction/semeval_internal/bernice/1707069752_jhu-clsp-bernice_dev_features.json \
 --classifier LogisticRegression