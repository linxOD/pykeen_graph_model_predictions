import os
import pathlib
import torch
from graph_model.graph_model import GraphModel

if __name__ == "__main__":
    # define saved model path
    # model_input_path = os.path.join("doctests", "test_pre_stratified_transe")

    # Add PosixPath to safe globals before loading the model
    torch.serialization.add_safe_globals([pathlib.PosixPath])

    # create an instance of the GraphModel class
    model = GraphModel(
        model_name="TransE",
        output_path="data")

    # create new dataset from ttl files
    source = os.path.join("data", "source")
    model.create_new_dataset(input_path=source)

    # train model
    model.train(epochs=100)

    # save model
    model.save_model()

    # load model results and create predicitons
    head = "https://sk.acdh.oeaw.ac.at/types/role/ANK"
    relation = "http://www.cidoc-crm.org/cidoc-crm/P94_has_created"
    model.predict_target(head=head, relation=relation)

    # save predictions as dataframe
    model.save_predictions_df()

    # visualize predictions
    model.visualize_predictions(top_n=10)
