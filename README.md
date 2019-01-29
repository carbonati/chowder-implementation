# CHOWDER Implementation

PyTorch implementation of [Classification and Disease Localization in Histopathology Using Only Global Labels: A Weakly-Supervised Approach](https://arxiv.org/pdf/1802.02212.pdf) published by Owkin, Inc.


## Documentation

CHOWDER is an approach for predicting general localized diseases in whole-slide-imaging by utilizing pre-trained deep convolutional networks, weakly-supervised learners, feature embeddings, and multiple instance learning via top instances and negative evidence scoring.

![image](https://github.com/carbonati/chowder-implementation/blob/master/png/metastasis_image.png)

### The Data
The data used for this specific study comes from the [Camelyon Challenge](https://camelyon16.grand-challenge.org/), which is a collection of whole-slide-images labeled as healthy or exhibits metastases. However, the WSI dataset provided by Owkin has already been preprocessed with the use of tissue detection, color normalization, and tiling. After those initial preprocessing steps each slide is then reduced to a uniform random sample of tiles each with a resolution of 224 x 224 pixels. After that, each tile is then transformed into a feature vector by using the tiles pre-output layer from  a pre-trained ResNet-50 architecture. Each feature vector consists of `P = 2048` floating point values, which is to be used as input for our CHOWDER network. These `P = 2048` floating point values are referred to as a patients ResNet features, which span from 40 to 1000 feature vectors for each patient in the dataset. 


### The network
The CHOWDER architecture consists of three modules (after the preprocessing steps are finished) - `feature embedding`, `multiple instance learning` (top instances and negative evidence), and a `multi-layer perceptron classifier`. 

![image](https://github.com/carbonati/chowder-implementation/blob/master/png/chowder_architecture.png)

#### Feature Embedding
A set of 1-D feature embeddings are given to each patient by using a one-dimensional convolutional layer whose kernel has the same dimensionality as each ResNet Feature, `P = 2048`. This embedding layer will also take on a stride of `P` and can be found in the class `chowder.model.ChowderArchitecture` as,
```python
self.embedding = nn.Conv1d(1, 1, 2048, stride=2048, padding=0)
```
where each patients set of ResNet features have been flatted into `N x P` dimensional vector, such that `N` is the number of tiles a given patient is sampled. The 1-D conv layer will slide across the flattened feature vector sharing weights across tiles.

#### Top Instances and Negative Evidence
After feature embedding, each patient will then have their N x 1 dimensional descriptor. The next step is for us to make use of the `MinMaxPoolingFunction` found in `chowder.minmax_pooling.py` where each descriptor is sorted in descending order
```python
# sort the feature embedding and save the max & min indices for backprop
x_sorted, x_indices = torch.sort(x_input.view(batch_size, n_regions), 
                                 dim=1, descending=True)
```

After sorting the output feature embedding we want to take top and bottom `R` entries, which tell us which regions have the most information and which regions support the absence of the class. We can grab the `R` max and min values with the following lines,

```python
self.indices_max = x_indices.narrow(1, 0, self.R)
self.indices_min = x_indices.narrow(1, -self.R, self.R)

output_max = x_sorted.narrow(1, 0, self.R)
output_min = x_sorted.narrow(1, -self.R, self.R)
```

It's important for us to store the indices used as the top and bottom instances since those are the only indices we will want to propogate back through the network when we are training. The last step for us will include concatenating the top and bottom instances together to pass as input for the MLP classifier,
```python
# concat the top & bottom instances for a MLP classifier (Figure 2)
output = torch.cat((output_max, output_min), dim=1)
```
after concatenating both outputs we will have a `2R x 1` output vector.


#### Multi-layer Perceptron (MLP) Classifier
The last piece of the architecture is to optimize the interactions between the top and bottom instances by passing them into a MLP with the form,

```python
self.classifier = nn.Sequential(
    nn.Linear(2*self.R, 200),
    nn.Sigmoid(),
    nn.Dropout(0.5),
    nn.Linear(200, 100),
    nn.Sigmoid(),
    nn.Dropout(0.5),
    nn.Linear(100, 1),
    nn.Dropout(0.5)
)
```

The final output is passed through one last sigmoid function and used for evaluation.

The code should be annotated thoroughly with comments to guide what important lines are accomplishing. 

### Usage

Here's a code snippet of how to train a CHOWDER network over a PyTorch data loader

```python
from chowder.model import ChowderArchitecture, ChowderModel
from chowder.maxmin_pooling import MaxMinPooling
from torch.utils.data import DataLoader

data_loader = DataLoader(torch_dataset, **kwargs)

num_instances = 5 # R
num_epochs = 10

maxmin_pooling = MaxMinPooling(R=num_instances)
chowder_model = ChowderModel(ChowderArchitecture, maxmin_pooling)

for epoch in range(num_epochs):
    chowder_model.fit(data_loader)
    y_pred = chowder_model.predict(data_loader)    
```

**Note**: the input data from `data_loader` passed into the network should be a torch tensor of shape `(batch_size, 1, P)`.

## Performance

Classification (AUC) results for the camelyon-16 dataset provided using `R`, the number of top and bottom instances to feed into the MLP classifier, and `E`, the number of ensemble CHOWDER networks, as the two hyper-parameters used to test performance.

CHOWDER |  CV (AUC) | Test (AUC)
-- | -- | -- |
(R = 1, E = 1) | 0.791 | 0.852 | 
(R = 2, E = 1) | 0.489 | 0.577 | 
(R = 5, E = 1) | 0.589 | 0.68 | 
(R = 10, E = 1) | 0.761 | 0.828 | 
(R = 1, E = 2) | 0.497 | 0.578 | 
(R = 2, E = 2) | 0.552 | 0.523 | 
(R = 5, E = 2) | 0.533 | 0.619 | 
(R = 10, E = 2) | 0.672 | 0.765 | 
(R = 1, E = 5) | 0.601 | 0.712 | 
(R = 2, E = 5) | 0.586 | 0.632 | 
(R = 5, E = 5) | 0.631 | 0.649 | 
(R = 10, E = 5) | 0.701 | 0.846 | 

I did not try any networks with `R` > 10 or `E` > 5 as the larger ensemble models started to be become computationally heavy to train. While running cross-validation the validation AUC experienced very high variance, which is likely due to the low amount of training data. There was also a degree of overfit, which I struggled with to overcome. 

## Requirements

Python 3.6+

All dependencies can be installed via `pip install -r requirements.txt`

* PyTorch 1.0.0
* Pandas
* Numpy
* Scipy
* scikit-learn

Make sure data.zip has been unzipped in the root directory via `unzip data.zip .`

## Reproducing the Experiment

### Training the CHOWDER model(s)

To train the CHOWDER model on the resnet features simply run `train.py` with 

`python3 train.py --train_data_path data/train_input/resnet_features --train_label_path data/train_output.csv --num_epochs 30 --num_splits 5 --num_instances 5`
#### Arguments
- `train_data_path`: Path to directory storing resnet features
    - default=data/train_input/resnet_features
- `train_label_path`: Path to .csv storing training labels
    - default=data/train_output.csv
- `num_epochs` (required): Number of epochs for each validation split
- `num_splits` (required): Number of splits for cross validation
- `num_instances` (required): Referred to as $R$ in the paper. The number of top & bottom instances to use after feature embedding
- `batch_size`: Batch size for training
    - default=10
- `save_model_dir`: Directory to save validation models if a path is given
    - default=None
- `num_workers`: Number of workers to speed up training time (4 cores on my beast local)
    - default=multiprocessing.cpu_count()

The script took roughly 1 hour to run using just 4 cores. If no `save_model_dir` is passed you can use the models found in the `models` directory for testing, otherwise each model will be saved to `save_model_dir` after training, which can then be used to predict test data using `test.py`

### Test Predictions
To predict the test data simply run `test.py` with 

`python3 test.py --test_data_path data/test_input/resnet_features --test_label_path data/test_output.csv --num_instances 5 --model_dir models --stack_models True`

#### Arguments
- `test_data_path`: Path to directory storing test resnet features
    - default=data/test_input/resnet_features
- `train_label_path`: Path to .csv with test labels
    - default=data/test_output.csv
- `num_instances` (required): Referred to as $R$ in paper paper. The number of top & bottom instances to use after feature embedding
- `model_dir`: Directory holding the CHOWDER model(s)
    - required
- `model_name`: Name of model to load from `model_dir`, but will be ignored if `stack_models = True`
    - default=''
- `stack_models`: Boolean whether to use all models in `model_dir` to predict on the test data and average the results
    - default=False
- `batch_size`: Batch size for training
    - default=10
- `num_workers`: Number of workers to speed up training time (4 cores on my beast local)
    - default=multiprocessing.cpu_count()

The above command will load in all CHOWDER models from `model_dir`, use each model to predict over the test data, then average the predictions from each model to output a final test prediction. The results should be saved to a pandas DataFrame as `test_chowder_5_instances.csv`. If you would like to use just one model you can leave `stack_models=False` and pass in the name of the model of your choice using `model_name`

Make sure when calling `test.py` that the `num_instances` passed is the same number of instances used to train the models being used!


## Limitations and potential improvements

A large and inefficient method found in this implementation is the use of 0 padding for every batch to ensure that each batch sample will fit the same number of dimensions. This is happening in the `chowder_collate` function where you'll see,
```python
# `lengths` are calculated as P * {max # of tiles in batch}
lengths = [features.shape[1] for features in resnet_features]
chowder_features = torch.zeros(len(targets), 1, max(lengths))   
for i, features in enumerate(resnet_features):
    feature_length = lengths[i]
    chowder_features[i, 0, :feature_length] = features[:feature_length]
```

There is likely a different approach that I'm not considering, but this seemed like the best fix to account for non-fixed sample sizes. Another issue this brings however, is the possibility that if there are no values < 0 in the feature embedding then the bottom instances will be 0, which will point to the `R` padded values. 


I struggled with both too much variability on the validation set along with overfitting the training set. I did not have enough time to effectively address these issues and validate if that the gradients passed through the network were computed correctly. 

An area of focus that I would improve if I were to spend more time with respect to the code is handling errors and raising exceptions. 

P.S.

Working for Owkin would mean everything to me!!!
