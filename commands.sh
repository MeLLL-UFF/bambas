#!/bin/bash

# sampling strategies
# semeval2015 = 0.3456
# semeval2016 = 0.7473
# semeval2024 = 0.4407

# ft extraction
python -m src.feature_extraction \
 --dataset semeval2016_paraphrased \
 --model jhu-clsp/bernice \
 --extraction_method cls \
 --output_dir semeval2016_paraphrased

python -m src.feature_extraction \
 --dataset semeval2015_paraphrased2_tweet_1to4 \
 --model jhu-clsp/bernice \
 --extraction_method cls \
 --output_dir semeval2015_paraphrased2_tweet_1to4


# run pure classification
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/semeval2024/train_features.json \
 --test_features feature_extraction/semeval2024/test_features.json \
 --dev_features feature_extraction/semeval2024/dev_features.json \
 --classifier LogisticRegression --binary --oversampling None

# run classification with ros
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/semeval2024/train_features.json \
 --test_features feature_extraction/semeval2024/test_features.json \
 --dev_features feature_extraction/semeval2024/dev_features.json \
 --classifier LogisticRegression --binary --oversampling RandomOverSampler --sampling_strategy 1

# run classification with smote
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/semeval2024/train_features.json \
 --test_features feature_extraction/semeval2024/test_features.json \
 --dev_features feature_extraction/semeval2024/dev_features.json \
 --classifier LogisticRegression --binary --oversampling SMOTE --sampling_strategy 1

# run classification on paraphrased dataset
python -m src.classification \
 --dataset paraphrase \
 --train_features feature_extraction/paraphrase/train_features.json \
 --test_features feature_extraction/paraphrase/test_features.json \
 --dev_features feature_extraction/paraphrase/dev_features.json \
 --classifier LogisticRegression --binary --oversampling None

python -m src.classification \
 --dataset semeval2015_paraphrased2_tweet_1to4 \
 --train_features feature_extraction/semeval2015_paraphrased2_tweet_1to4/train_features.json \
 --test_features feature_extraction/semeval2015_paraphrased2_tweet_1to4/test_features.json \
 --dev_features feature_extraction/semeval2015_paraphrased2_tweet_1to4/dev_features.json \
 --classifier LogisticRegression --binary --oversampling None