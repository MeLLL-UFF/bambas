for i in $(seq 1 10); do
    python -m src.classification \
    --dataset semeval_internal \
    --train_features feature_extraction/1706729778_jhu-clsp-bernice_train_features.json \
    --test_features feature_extraction/1706729778_jhu-clsp-bernice_test_features.json \
    --dev_features feature_extraction/1706729778_jhu-clsp-bernice_dev_features.json \
    --classifier LogisticRegression \
    --seed "$i"
done