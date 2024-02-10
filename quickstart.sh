# ft extraction using bernice with cls token
python -m src.feature_extraction \
 --dataset semeval2024 \
 --model jhu-clsp/bernice \
 --extraction_method cls \
 --output_dir test_folder/ 

# classification using LogisticRegression with Binary Relevance + Combined Oversampling
python -m src.classification \
 --dataset semeval2024 \
 --train_features feature_extraction/test_folder/train_features.json \
 --test_features feature_extraction/test_folder/test_features.json \
 --dev_features feature_extraction/test_folder/dev_features.json \
 --classifier LogisticRegression \
 --oversampling Combination