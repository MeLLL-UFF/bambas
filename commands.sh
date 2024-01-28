python -m src.feature_extraction \
 --dataset semeval_augmented \
 --model xlm-roberta-base \
 --extraction_method cls

# augmented label preserving
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706474866_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706474866_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706474866_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# augmented
python -m src.classification \
 --dataset semeval_augmented \
 --train_features feature_extraction/1706466400_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1706466400_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1706466400_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression

# semeval2024
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier RidgeClassifier