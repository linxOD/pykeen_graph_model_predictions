# import os
# import glob
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
        output_path="data",
        training_path="data/dataset/training.tsv",
        testing_path="data/dataset/testing.tsv",
        evaluation_path="data/dataset/evaluation.tsv")

    # # # convert ttl to tsv
    # model.convert_ttl_to_tsv(input_glob=glob.glob("data/source/*.ttl"))

    # model.combine_tsv(glob.glob("data/source/*.tsv"),
    #                   combined_path="data/combined-graph.tsv")

    # model.randomize_split_tsv(combined_path="data/combined-graph.tsv")

    # # train model
    # model.train(epochs=100)

    # save model
    # model.save_model()

    # load model results and create predicitons
    head = "https://sk.acdh.oeaw.ac.at/types/role/ANK"
    relation = "http://www.cidoc-crm.org/cidoc-crm/P94_has_created"
    model.predict_target(head=head, relation=relation)

    # save predictions as dataframe
    model.save_predictions_df()

    # visualize predictions
    model.visualize_predictions(top_n=10)
