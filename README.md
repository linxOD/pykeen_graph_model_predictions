# Pykeen GraphModel Pipeline
This repository contains the code for the Pykeen GraphModel pipeline. The pipeline is designed to train and evaluate a Pykeen model on a given dataset

## How To Use
To use the pipeline, you need to have a dataset in the form of a .tsv file. The dataset should have three columns: subject(head), predicate(relation), and object(tail). The pipeline will train a Pykeen model on this dataset and predict tails for a give head and relation. Model evaluation follows...

### Run Pipeline
Update the GraphModel arguments and run the python script in the console. E.g. `pipeline.py`

Class:
* GraphModel(**kwargs)
    * model_name (str of pykeen model classes)
    * model_output_path (str with posix path to save model)
    * training_path (str with posix path to training data)
    * testing_path (str with posix path to testing data)
    * evaluation_path (str with posix path to evaluation data)

Methods:
* convert_ttl_to_tsv(**kwargs)
    * ttl_path (str with posix path)
    * tsv_path (str with posix path)
* train(**kwargs)
    * epochs (int e.g. 100)
* save_model()
* predict_targets(**kwargs)
    * head (str URI)
    * relation (str URI)
* visualize_predictions(**kwargs)
    * graph_output_path (str with posix path)
    * top_n (int with max number of predictions to visualize)