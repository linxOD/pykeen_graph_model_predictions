import torch
import os
import pathlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import rdflib

from pydantic import BaseModel
from pykeen.pipeline import pipeline
from tqdm import tqdm
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target


class GraphModel(BaseModel):

    # variables for model
    model_name: str = "TransE"
    model_output_path: str = os.path.join(
        "model",
        f"semantic_kraus_experiment_{model_name}")

    ########################################
    # variables for training and prediction
    ########################################

    # set os paths to the data as tsv files
    training_path: str = "data/lk-texts.tsv"
    testing_path: str = "data/fa-texts.tsv"
    evaluation_path: str = "data/dw-texts.tsv"

    # training, testing and validation data
    # will be loaded as TriplesFactory objects
    # by calling the load_training_data, load_testing_data
    # and load_evaluation_data methods
    # or passed directly as TriplesFactory objects
    training: dict = None
    testing: dict = None
    validation: dict = None
    model_results: dict = None

    # prediction targets are stored as a list of tuples
    # by calling the method predict_target
    prediction_targets: list = []

    def convert_ttl_to_tsv(self, ttl_path=None, tsv_path=None):
        print(f"""Loading {ttl_path} file and
              converting to {tsv_path}...""")
        # Load the TTL file
        g = rdflib.Graph()
        g.parse(ttl_path, format="turtle")

        # Extract triples
        triples = []
        for s, p, o in g:
            # Convert URIs to strings and remove angle brackets
            subject = str(s).strip('<>')
            predicate = str(p).strip('<>')
            obj = str(o).strip('<>')

            triples.append((subject, predicate, obj))

        # Create DataFrame and save as TSV
        df = pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Conversion complete. File saved as {tsv_path}")

        return tsv_path

    def load_training_data(self):
        print(f"Loading training data from {self.training_path}...")
        return TriplesFactory.from_path(self.training_path,
                                        create_inverse_triples=True)

    def load_testing_data(self):
        print(f"Loading testing data from {self.testing_path}...")
        return TriplesFactory.from_path(
            self.testing_path,
            entity_to_id=self.training.entity_to_id,
            relation_to_id=self.training.relation_to_id,
            create_inverse_triples=True)

    def load_evaluation_data(self):
        print(f"Loading testing data from {self.evaluation_path}...")
        return TriplesFactory.from_path(
            self.evaluation_path,
            entity_to_id=self.training.entity_to_id,
            relation_to_id=self.training.relation_to_id,
            create_inverse_triples=True)

    def train(self, epochs: int = 5):
        print("Hello from pykeen-kraus-experiment!")

        self.training = self.load_training_data()
        self.testing = self.load_testing_data()

        self.model_results = pipeline(
            training=self.training,
            testing=self.testing,
            model=self.model_name,
            epochs=epochs)

        return self.model_results

    def save_model(self):
        print(f"Saving model to {self.model_output_path}...")
        results = self.results
        results.save_to_directory(self.model_output_path)
        print("Model saved.")

    def predict_target(self, head, relation):
        print("Hello from pykeen-kraus-experiment!")

        self.training = self.load_training_data()
        self.testing = self.load_testing_data()
        self.validation = self.load_evaluation_data()

        # Add PosixPath to safe globals before loading the model
        torch.serialization.add_safe_globals([pathlib.PosixPath])

        # load the model
        if self.model_results is None:
            self.model_results = torch.load(
                os.path.join(self.model_output_path, 'trained_model.pkl'),
                map_location=torch.device('cpu'),
                weights_only=False)

        print("model loaded...")

        print("predicting targets...")
        # create predicitons
        predictions = predict_target(
            model=self.model_results,
            triples_factory=self.training,
            head=head,
            relation=relation,
        )

        # filter predictions
        pred_filered = predictions.filter_triples(self.training)

        # add membership columns
        pred_annotated = pred_filered.add_membership_columns(
            validation=self.validation,
            testing=self.testing)

        # get the dataframe
        pred_filtered_df = pred_annotated.df

        # create new triples
        self.prediction_targets = []
        self.prediction_targets += [(head,
                                    relation,
                                    row['tail_label'],
                                    row['score'])
                                    for _, row in pred_filtered_df.iterrows()]

        return self.prediction_targets

    def visualize_predictions(self,
                              top_n: int = None,
                              graph_output_path: str = None):
        print("Visualizing predictions...")

        # create a directed graph
        G = nx.DiGraph()

        # reduce inputs
        if top_n is not None:
            self.prediction_targets = self.prediction_targets[:top_n]

        # add nodes and edges
        for head, relation, tail, score in tqdm(
                self.prediction_targets,
                total=len(self.prediction_targets)):

            G.add_node(head, color='lightblue')
            G.add_node(tail, color='lightgreen')
            G.add_edge(head, tail, label=relation, weight=score)

        # draw the graph
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G,
                                     pos,
                                     edge_labels=edge_labels,
                                     font_size=6)
        nx.draw(G,
                with_labels=True,
                font_size=8,
                node_size=500)
        plt.title("Predicted Targets Graph")
        plt.axis('off')
        plt.tight_layout()

        # saving the graph as png
        print("Saving visualization as prediction_targets_graph.png...")
        os.makedirs(graph_output_path, exist_ok=True)
        plt.savefig(os.path.join(
            graph_output_path, 'prediction_targets_graph.png'),
                    dpi=300)

        return graph_output_path
