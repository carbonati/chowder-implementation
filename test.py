import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from dataset import ChowderDataset, chowder_collate
from chowder.model import ChowderArchitecture, ChowderModel
from chowder.maxmin_pooling import MaxMinPooling
import argparse
import datetime
import multiprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', default='data/test_input/resnet_features',
                        help='Path to directory storing test resnet features')
    parser.add_argument('--test_label_path', default='data/test_output.csv',
                        help='Path to .csv with test IDs')
    parser.add_argument('--data_dir', default='data', 
                        help='main directory to store data files')
    parser.add_argument('--num_instances', required=True, type=int,
                        help="Refered to as R in paper. The number of top & bottom \
                        instances to use after feature embeding")
    parser.add_argument('--model_dir', required=True,
                        help='Directory holding the CHOWDER model(s)')
    parser.add_argument('--model_name', default='', 
                        help='Name of model to load from `model_dir`, but will be \
                        ignored if `stack_models` = True')
    parser.add_argument('--stack_models', default=False, type=bool,
                        help='Boolean whether to use all models in `model_dir` \
                        to predict on the test data and average the results')
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='Batch size for prediction')
    parser.add_argument('--num_workers', default=multiprocessing.cpu_count(),
                        help='4 core beast mode')   

    args = parser.parse_args()

    # MUST use the same R (`num_instances`) used for training to load model weights
    df_test = pd.read_csv(args.test_label_path)
    test_ids = df_test['ID'].values

    test_dataset = ChowderDataset(args.test_label_path, args.test_data_path, test_ids)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                         num_workers=args.num_workers,collate_fn=chowder_collate)

    maxmin_pooling = MaxMinPooling(R=args.num_instances)
    chowder_model = ChowderModel(ChowderArchitecture, maxmin_pooling)

    if args.stack_models:
        preds = []
        model_filenames = os.listdir(args.model_dir)
        for model_name in model_filenames:
            # don't need to reinstanciate a new model to load new weights!
            try:
                chowder_model.load(os.path.join(args.model_dir, model_name))
            except RuntimeError:
                print('Make sure `--num_instances` is the same R as the model \
                       being used was trained with!')
                raise 
            y_preds = chowder_model.predict(test_dl)
            preds.append(y_preds)
        # average the predictions (ensemble modeling)
        preds_test = np.mean(preds, axis=0)
    else:
        chowder_model.load(os.path.join(args.model_dir, args.model_name))
        preds_test = chowder_model.predict(test_dl)
        model_filenames = [args.model_name]

    # save the test results to disk
    test_output = pd.DataFrame({"ID": test_ids, "Target": preds_test[:,0]})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(os.path.join(args.data_dir, "test_chowder_{}_instances_{}_ensemble.csv").format(
        args.num_instances, len(model_filenames)))