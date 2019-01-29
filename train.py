import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from chowder.dataset import ChowderDataset, chowder_collate
from chowder.model import ChowderArchitecture, ChowderModel
from chowder.maxmin_pooling import MaxMinPooling
import argparse
import datetime
import multiprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='data/train_input/resnet_features',
                        help='Path to directory storing resnet features')
    parser.add_argument('--train_label_path', default='data/train_output.csv',
                        help='Path to .csv storing training labels')
    parser.add_argument('--num_epochs', required=True, type=int,
                        help='Numer of epochs for each validation split')
    parser.add_argument('--num_splits', required=True, type=int,
                        help='Number of splits for cross validation')
    parser.add_argument('--num_instances', required=True, type=int,
                        help="Referred to as R in paper. The number of top & bottom \
                        instances to use after feature embeding")
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='Batch size for training')
    parser.add_argument('--save_model_dir', default=None,
                        help='Directory to save validation models if a path is given')
    parser.add_argument('--num_workers', default=multiprocessing.cpu_count(),
                        help='4 core beast mode')   

    args = parser.parse_args()

    folder = StratifiedKFold(n_splits=args.num_splits)
    aucs = []

    # if a model path is saved create it if it does not exist
    if args.save_model_dir:
        if not os.path.exists(args.save_model_dir):
            os.mkdir(args.save_model_dir)

    # read in full training data for validatoin splits
    df_full = pd.read_csv(args.train_label_path)

    # cross validation trainer
    for split_id, (tr_ind, val_ind) in enumerate(folder.split(df_full['ID'], df_full['Target'])):
        print("Starting split {0}/{1} @ {2}".format(split_id+1, args.num_splits,
            datetime.datetime.now().strftime("%H:%m:%S")))
        # split data into train and validation
        train_ids = df_full.iloc[tr_ind]['ID']
        val_ids = df_full.iloc[val_ind]['ID']
        
        train_dataset = ChowderDataset(args.train_label_path, args.train_data_path, train_ids)
        val_dataset = ChowderDataset(args.train_label_path, args.train_data_path, val_ids)
        
        train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers = args.num_workers, collate_fn=chowder_collate)
        val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers = args.num_workers, collate_fn=chowder_collate) 
        y_val = np.array(val_dataset.get_labels())
        
        # instantiate a new CHOWDER model each split
        maxmin_pooling = MaxMinPooling(R=args.num_instances)
        chowder_model = ChowderModel(ChowderArchitecture, maxmin_pooling)
        

        total_cv_auc = 0

        for epoch in range(args.num_epochs):
            chowder_model.fit(train_dl)
            y_pred = chowder_model.predict(val_dl)
            
            val_loss = log_loss(y_val, y_pred)
            val_auc = roc_auc_score(y_val, y_pred)
            
            print("\nEpoch {0}: Val AUC: {1:.6f}\tVal Loss: {2:.6f}\t\n".format(
                epoch+1, val_auc, val_loss))
            total_cv_auc += val_auc
        
        # record AUC for each split to compute mean AUC
        aucs.append(total_cv_auc / args.num_epochs)

        # save each CV model to predict on test data
        if args.save_model_dir:
            model_filepath = os.path.join(args.save_model_dir, 'chowder_{0}_instances_model_{1}.pth'.format(
                args.num_instances, split_id+1))
            chowder_model.save(model_filepath)
            print("Saved model to {}\n".format(model_filepath))
            
    print("Predicting weak labels by CHOWDER (R = {})".format(args.num_instances))
    print("AUC ({0} Splits): mean {1}, std {2}".format(args.num_splits, np.mean(aucs), np.std(aucs)))