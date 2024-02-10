for i in 9 8 7 6 5 4 3 2 1; do
    python -m src.classification \
    --dataset semeval2024 \
    --train_features feature_extraction/1701981179_xlm-roberta-base_train_features.json \
    --test_features feature_extraction/1701981179_xlm-roberta-base_test_features.json \
    --dev_features feature_extraction/1701981179_xlm-roberta-base_dev_features.json \
    --classifier LogisticRegression \
    --seed 10 \
    --sampling_strategy 0."$i";
    # echo 0."$i"
done