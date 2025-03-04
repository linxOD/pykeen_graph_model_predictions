import os
from graph_model.graph_model import GraphModel

if __name__ == "__main__":
    # define saved model path
    model_input_path = os.path.join("doctests", "test_pre_stratified_transe")

    # create an instance of the GraphModel class
    model = GraphModel(
        model_output_path=model_input_path,
        training_path="data/lk-texts.tsv",
        testing_path="data/fa-texts.tsv",
        evaluation_path="data/dw-texts.tsv")

    # convert ttl to tsv
    # model.convert_ttl_to_tsv(ttl_path="data/lk-texts.ttl",
    #                          tsv_path="data/lk-texts.tsv")
    # model.convert_ttl_to_tsv(ttl_path="data/fa-texts.ttl",
    #                          tsv_path="data/fa-texts.tsv")
    # model.convert_ttl_to_tsv(ttl_path="data/dw-texts.ttl",
    #                          tsv_path="data/dw-texts.tsv")

    # train model
    # model.train(epochs=5)

    # save model
    # model.save_model()

    # load model results and create predicitons
    head = "https://sk.acdh.oeaw.ac.at/types/role/ANK"
    relation = "http://www.cidoc-crm.org/cidoc-crm/P94_has_created"
    model.predict_target(head=head, relation=relation)

    # visualize predictions
    model.visualize_predictions(
        graph_output_path="graph_output",
        top_n=10)
