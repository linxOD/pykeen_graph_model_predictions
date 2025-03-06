import torch
import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import rdflib
import glob

from datetime import datetime
from pydantic import BaseModel
from pykeen.pipeline import pipeline
from tqdm import tqdm
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target


class GraphModel(BaseModel):

    # variables for model
    model_name: str = "TransE"
    output_path: str = "data"

    ########################################
    # variables for training and prediction
    ########################################

    # set os paths to the data as tsv files
    training_path: str = os.path.join(output_path, "dataset", "training.tsv")
    testing_path: str = os.path.join(output_path, "dataset", "testing.tsv")
    evaluation_path: str = os.path.join(output_path, "dataset",
                                        "evaluation.tsv")

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

    def convert_ttl_to_tsv(self, input_path: str):
        input_glob = glob.glob(os.path.join(input_path, "*.ttl"))
        for ttl_path in tqdm(input_glob, total=len(input_glob)):
            tsv_path = ttl_path.replace('.ttl', '.tsv')
            print(f"""Loading {ttl_path} file and
                converting to {tsv_path}...\n\n""")
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
            df = pd.DataFrame(triples,
                              columns=['subject', 'predicate', 'object'])
            df.to_csv(tsv_path, sep='\t', index=False)
            print(f"Conversion complete. File saved as {tsv_path}")
        print("Conversion to tsv complete.\n\n")

    # create method to compine the training and testing data and
    # evaluate in one tsv
    def combine_tsv(self, input_path: str):
        input_glob = glob.glob(os.path.join(input_path, "*.tsv"))
        combined = pd.DataFrame(columns=['subject', 'predicate', 'object'])
        combined_path = os.path.join("data", "combined-graph.tsv")

        for tsv_path in tqdm(input_glob, total=len(input_glob)):
            print(f"""Loading {tsv_path} and
                combining to {combined_path}...\n\n""")
            # Load the training and testing and evaluation files
            doc = pd.read_csv(tsv_path, sep='\t')

            # Combine the training and testing and evaluation data
            combined = pd.concat([combined, doc], ignore_index=True)

        # Save the combined data as TSV
        combined.to_csv(combined_path, sep='\t', index=False)
        print(f"Combining complete. File saved as {combined_path}\n\n")
        return combined_path

    # create method to randomize and split tsv into training, testing and
    # evaluation data
    def randomize_split_tsv(self, combined_path: str):

        print(f"""Loading {combined_path} file and
              randomizing and dataset/splitting to dataset/training.tsv,
              dataset/testing.tsv and evauluation.tsv ...""")
        # Load the combined file
        combined = pd.read_csv(combined_path, sep='\t')

        # Randomize the combined data
        combined = combined.sample(frac=1).reset_index(drop=True)

        # Split the combined data into training and testing and evaluation data
        training = combined.iloc[:int(len(combined)*0.6)]
        testing = combined.iloc[int(len(combined)*0.6):int(len(combined)*0.8)]
        evaluation = combined.iloc[int(len(combined)*0.8):]

        # Save the training and testing and evaluation data as TSV
        os.makedirs(os.path.join(self.output_path, "dataset"), exist_ok=True)
        self.training_path = os.path.join(self.output_path,
                                          "dataset",
                                          "training.tsv")

        self.testing_path = os.path.join(self.output_path,
                                         "dataset",
                                         "testing.tsv")

        self.evaluation_path = os.path.join(self.output_path,
                                            "dataset",
                                            "evaluation.tsv")

        training.to_csv(self.training_path, sep='\t', index=False)
        testing.to_csv(self.testing_path, sep='\t', index=False)
        evaluation.to_csv(self.evaluation_path, sep='\t', index=False)

        # remove combined file
        os.remove(combined_path)
        print("Complete. Files saved in data/dataset folder.\n\n")

    def create_new_dataset(self, input_path: list):
        os.makedirs(self.output_path, exist_ok=True)
        # convert ttl to tsv
        self.convert_ttl_to_tsv(input_path)
        # combine tsv
        combined_graph = self.combine_tsv(input_path)
        # randomize and split tsv
        self.randomize_split_tsv(combined_graph)

    def load_create_triples_factory(self, type: str = "training"):
        if type == "testing":
            input_path = self.testing_path
        elif type == "evaluation":
            input_path = self.evaluation_path
        elif type == "training":
            input_path = self.training_path
        else:
            raise ValueError("""Invalid type. Choose from 'testing',
                             'evaluation' or 'training'""")
        print(f"Loading {type} data from {input_path}...\n\n")
        if type == "training":
            triples_factory = TriplesFactory.from_path(
                input_path,
                create_inverse_triples=True)
        else:
            triples_factory = TriplesFactory.from_path(
                input_path,
                entity_to_id=self.training.entity_to_id,
                relation_to_id=self.training.relation_to_id,
                create_inverse_triples=True)
        return triples_factory

    def train(self, epochs: int = 5):
        print("\n\nHello from pykeen-kraus-experiment!\n\n")
        os.makedirs(self.output_path, exist_ok=True)

        self.training = self.load_create_triples_factory(type="training")
        self.testing = self.load_create_triples_factory(type="testing")

        model_output_path = os.path.join(self.output_path, 'model')

        self.model_results = pipeline(
            training=self.training,
            testing=self.testing,
            model=self.model_name,
            epochs=epochs,
            device=torch.device('cuda'
                                if torch.cuda.is_available() else 'cpu'),
            training_kwargs=dict(
                num_epochs=100,
                checkpoint_name='my_checkpoint.pt',
                checkpoint_directory=model_output_path,
                checkpoint_frequency=5,
                checkpoint_on_failure=True
            ),
            random_seed=1235
        )

        return self.model_results

    def save_model(self):
        model_output_path = os.path.join(self.output_path, 'model')
        os.makedirs(model_output_path, exist_ok=True)
        print(f"Saving model to {self.model_output_path}...")
        results = self.model_results
        results.save_to_directory(self.model_output_path)
        print("Model saved.\n\n")

    def predict_target(self, head, relation):
        print("\n\nHello from pykeen-kraus-experiment!\n\n")

        self.training = self.load_create_triples_factory(type="training")
        self.testing = self.load_create_triples_factory(type="testing")
        self.validation = self.load_create_triples_factory(type="evaluation")

        model_output_path = os.path.join(self.output_path, 'model')

        # load the model
        if self.model_results is None:
            self.model_results = torch.load(
                os.path.join(model_output_path, 'trained_model.pkl'),
                map_location=torch.device('cuda'
                                          if torch.cuda.is_available()
                                          else 'cpu'),
                weights_only=False)

        print("model loaded...")

        print("predicting targets...\n\n")
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
        pred_filtered_df = pred_annotated.df.sort_values(by='score',
                                                         ascending=False)

        # create new triples
        self.prediction_targets = []
        self.prediction_targets += [(head,
                                    relation,
                                    row['tail_label'],
                                    row['score'])
                                    for _, row in pred_filtered_df.iterrows()]

        return self.prediction_targets

    def save_predictions_df(self):
        print("Saving predictions as dataframe...")
        os.makedirs(self.output_path, exist_ok=True)
        # create a dataframe
        df = pd.DataFrame(self.prediction_targets,
                          columns=['head', 'relation', 'tail', 'score'])
        # save the dataframe
        new_date = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"prediction_targets_{new_date}.csv"
        prediciton_output_path = os.path.join(self.output_path, "predictions")
        os.makedirs(prediciton_output_path, exist_ok=True)
        df.to_csv(os.path.join(prediciton_output_path, filename), index=False)

        print(f"""Dataframe saved as {filename} in
              {prediciton_output_path}.\n\n""")
        return prediciton_output_path

    def visualize_predictions(self, top_n: int = None):
        print("Visualizing predictions...")
        os.makedirs(self.output_path, exist_ok=True)

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
        nx.draw(G, with_labels=True, font_size=6)
        plt.title("Predicted Targets Graph")
        plt.axis('off')
        plt.tight_layout()

        # saving the graph as png
        graph_output_path = os.path.join(self.output_path, "graph")
        print("Saving visualization as prediction_targets_graph.png..\n\n")
        os.makedirs(graph_output_path, exist_ok=True)
        new_date = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"""prediction_targets_graph_{new_date}.png"""
        plt.savefig(os.path.join(
            graph_output_path, filename),
                    dpi=300)

        return graph_output_path
