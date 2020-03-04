import os
import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from gp_comp.example.classes.chain import Chain
from gp_comp.example.classes.model import *
from gp_comp.example.classes.node import NodeGenerator
from gp_comp.example.classes.utils import project_root
from gp_comp.example.classes.visualisation import ChainVisualiser

random.seed(1)
np.random.seed(1)


def calculate_validation_metric(chain: Chain, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = chain.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


# the dataset was obtained from https://www.kaggle.com/kashnitsky/a5-demo-logit-and-rf-for-credit-scoring

# a dataset that will be used as a train and test set during composition
file_path_train = 'data\scoring_train.csv'
full_path_train = os.path.join(str(project_root()), file_path_train)
dataset_to_compose = InputData.from_csv(full_path_train)

# a dataset for a final validation of the composed model
file_path_test = 'data\scoring_test.csv'
full_path_test = os.path.join(str(project_root()), file_path_test)
dataset_to_validate = InputData.from_csv(full_path_test)

# start chain building
new_chain = Chain()

last_node = NodeGenerator.secondary_node(MLP())

y1 = NodeGenerator.primary_node(XGBoost(), dataset_to_compose)
new_chain.add_node(y1)

y2 = NodeGenerator.primary_node(LDA(), dataset_to_compose)
new_chain.add_node(y2)

y3 = NodeGenerator.secondary_node(XGBoost(), [y1, y2])
new_chain.add_node(y3)

y4 = NodeGenerator.primary_node(KNN(), dataset_to_compose)
new_chain.add_node(y4)
y5 = NodeGenerator.primary_node(XGBoost(), dataset_to_compose)
new_chain.add_node(y5)

y6 = NodeGenerator.secondary_node(XGBoost(), [y4, y5])
new_chain.add_node(y6)

last_node.nodes_from = [y3, y6]
new_chain.add_node(last_node)

visualiser = ChainVisualiser()
visualiser.visualise(new_chain)

# the quality assessment for the obtained composite model
roc_on_chain = calculate_validation_metric(new_chain, dataset_to_validate)

print(f'ROC AUC is {round(roc_on_chain, 3)}')
