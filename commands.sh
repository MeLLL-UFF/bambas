#!/bin/bash

# ft extraction using semeval base
python -m src.feature_extraction \
 --dataset semeval_internal \
 --model jhu-clsp/bernice \
 --extraction_method cls

# semeval_internal
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/semeval2024/bernice/1707079469_jhu-clsp-bernice_train_features.json \
 --test_features feature_extraction/semeval2024/bernice/1707079469_jhu-clsp-bernice_test_features.json \
 --dev_features feature_extraction/semeval2024/bernice/1707079469_jhu-clsp-bernice_dev_features.json \
 --classifier LogisticRegression

# semeval_internal trained with train test dev
python -m src.classification \
 --dataset semeval_internal \
 --train_features feature_extraction/1706826974_jhu-clsp-bernice_train_features.json \
 --test_features feature_extraction/1706826974_jhu-clsp-bernice_test_features.json \
 --dev_features feature_extraction/1706826974_jhu-clsp-bernice_dev_features.json \
 --classifier LogisticRegression

# ft extraction semeval augmented
python -m src.feature_extraction \
 --dataset semeval_augmented \
 --model xlm-roberta-base \
 --extraction_method cls

 
# semeval2024
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# using ft from bernice
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/1706662916_jhu-clsp-bernice_train_features.json \
 --test_features feature_extraction/1706662916_jhu-clsp-bernice_test_features.json \
 --dev_features feature_extraction/1706662916_jhu-clsp-bernice_dev_features.json \
 --classifier LogisticRegression

# using ft from bertweet-base
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/1706660251_vinai-bertweet-base_train_features.json \
 --test_features feature_extraction/1706660251_vinai-bertweet-base_test_features.json \
 --dev_features feature_extraction/1706660251_vinai-bertweet-base_dev_features.json \
 --classifier LogisticRegression

# augmented from ptc reductio
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706560674_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706560674_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706560674_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# augmented from 2301
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706492805_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706492805_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706492805_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# augmented label preserving Reductio ad Hitlerum
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706486057_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706486057_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706486057_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# augmented Reductio ad Hitlerum
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706482884_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706482884_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706482884_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# augmented label preserving Smears
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706474866_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706474866_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706474866_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# augmented Smears
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706466400_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706466400_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706466400_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression
