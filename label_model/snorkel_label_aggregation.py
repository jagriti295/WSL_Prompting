import logging
import torch
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.labelmodel import Snorkel
from wrench.endmodel import EndClassifierModel
import json
from collections import defaultdict

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cpu')

#### Load dataset
dataset_path = './datasets/'
data = 'agnews'

# the agnews dataset needs to have train, valid and test .json files
# the number of weak labels must be same for all 3 files e.g. 9 for LF-based weak labels and 10 after adding the LLM annotations
# modify the agnews folder to have the correct schema and dataset size, and weak labels. Keeping the entire dataset takes too long to run
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    # device=device,
    extract_fn='bert', # extract bert embedding
    model_name='bert-base-cased',
    cache_name='bert'
)

#### Run label model: Snorkel
label_model = Snorkel(
    lr=0.01,
    l2=0.0,
    n_epochs=10
)
label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)


#### Filter out uncovered training data - only keep training instances that have at least one weak label
train_data = train_data.get_covered_subset()
aggregated_hard_labels = label_model.predict(train_data)
aggregated_soft_labels = label_model.predict_proba(train_data)
# print("aggregated_hard_labels:", aggregated_hard_labels.shape)
# print("aggregated_soft_labels:", aggregated_soft_labels.shape)
# print("agg:", aggregated_soft_labels[0])

acc = label_model.test(test_data, 'acc')
logger.info(f'label model test acc: {acc}')

acc = label_model.test(train_data, 'acc')
logger.info(f'label model train acc: {acc}')

ids = []
for i in range(len(train_data.ids)):
    ids.append(train_data.ids[i])

# save the training ids of the covered training instances - this is used to load the training data in the next step of BERT finetuning
with open("./datasets/agnews/train_out_ids.txt", "w") as f:
    f.write(" ".join(ids))

# save the aggregated labels to be used during BERT finetuning
torch.save(aggregated_hard_labels, './datasets/agnews/aggregated_hard_labels.pt')
torch.save(aggregated_soft_labels, './datasets/agnews/aggregated_soft_labels.pt')