### Script Information:
Data Preparation 
----------
1. data_extraction --> Collects all stream dataset from Kaggle and stores in ~/data/raw folder
2. fod_data_structuring --> Converts FOD-A Pascal-VOC to YOLO format & Splits into Train/Valid/Test subsets
3. generate_yaml_config --> Generates YAML configuration files for all streams
4. store_processed_data --> Copies the processed Turnaround, PPE, & FOD dataset into ~/data/processed folder
5. eda_pipeline --> Analyzes Datasets, prints dataset summary tables & shows visualization
6. directory_structure --> Prints the directory structure for a given path (Helper Function)
7. yaml_reader --> Read & extract metadate from yaml config files (Helper Function)


### Notebook Information:
1. Data Preparation Pipeline
Data Extraction --> FOD Conversion & Split --> YAMl file generation --> Data Structuting --> Data Visualization
