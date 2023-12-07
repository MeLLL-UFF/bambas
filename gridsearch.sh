python -m src.classification \
--dataset semeval2024 \
--train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier RidgeClassifier
 
 python -m src.classification \
--dataset semeval2024 \
--train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier RandomForestClassifier
 
 python -m src.classification \
--dataset semeval2024 \
--train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier ExtraTreesClassifier
 
 python -m src.classification \
--dataset semeval2024 \
--train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier KNeighborsClassifier
 
 python -m src.classification \
--dataset semeval2024 \
--train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier RadiusNeighborsClassifier
 
 python -m src.classification \
--dataset semeval2024 \
--train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
 --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
 --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
 --classifier LogisticRegression