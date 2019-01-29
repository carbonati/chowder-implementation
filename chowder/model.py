import numpy as np
import torch
import torch.nn as nn
from .utils import MeanMetric
from sklearn.metrics import roc_auc_score 
import time, datetime


class ChowderArchitecture(nn.Module):
    """CHOWDER architecture described in section 2.3 (Figure 2)"""
    
    def __init__(self, pooling):
        """
            Args:
                pooling: pytorch pooling submodule
        """
        super().__init__()
        
        # 1D conv layer used as the "same embedding for every tile"
        # Since each resnet feature is flattened we can run the conv layer
        # with kernal of shape P and stride P will allow us to share weights
        # across tiles 
        self.embedding = nn.Conv1d(1, 1, 2048, stride=2048, padding=0)
        self.spatial_pool = pooling
        self.R = pooling.R
        
        # MLP classifier 
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
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        """Propogate forward through the network
        
            Args:
                x: input torch tensor of shape (batch_size, 1, P*N_{tiles})
        """
        x = self.embedding(x)
        x = self.spatial_pool(x)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x        
    
    
    def save_model(self, fn):
        """Helper function to save a CHOWDER model parameters"""
        torch.save(self.state_dict(), fn)
    
    
    def load_model(self, fn):
        """Helper function to load a CHOWDER model parameters"""
        state_dict = torch.load(fn, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)



class ChowderModel(nn.Module):
    """Chowder model to train on patients resnet features"""
    def __init__(self, model, pooling, learning_rate=0.001):
        """
            Args:
                model: Chowder archicture 
                pooling: pytorch pooling submodule
                learning_rate: learning rate (default: 0.001)
        """
        super().__init__()

        # set up optimization criteria for training (section 3.1)
        self.model = model(pooling)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate)
        self.optimizer.zero_grad()
        
    
    def set_train_mode(self):
        """Set model to training model for learning"""
        self.model.train()
    
    
    def set_predict_mode(self):
        """Set model to evaluation mode for prediction"""
        self.model.eval()
   

    def fit(self, data_loader, verbose_iter=10):
        """Fits the CHOWDER model to the Camelyon-16 data and prints
           loss & accuracy metrics every `verbose_iter` batches
        
            Args:
                data_loader: pytorch Data Loader to iterate over batches
                verbose_iter: int to determine when to print training progress
        """
        self.set_train_mode() # set model to train mode for learning
        losses = MeanMetric()
        preds = []
        targets = []
        batches_per_epoch = len(data_loader) - 1
        
        start_time = time.time()
        for batch_num, (X, y) in enumerate(data_loader):
            y_pred = self.model(X)[:,0]
            loss = self.criterion(y_pred, y)
            
            # numpy conversion for efficient memory storage
            y_pred = y_pred.cpu().data.numpy()
            y = y.cpu().numpy()
            
            # standard pytorch backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # save results for loggin
            losses.update(loss.item(), len(y))
            preds.extend(y_pred)
            targets.extend(y)
            
            if batch_num % verbose_iter == 0:
                elapsed_time = round(time.time() - start_time)
                elapsed_time = datetime.timedelta(seconds=elapsed_time)
                
                auc_score = roc_auc_score(targets, preds)
                print("[{0}] Train step {1}/{2}\tLoss: {3:.6f}\tAUC: {4:.6f}".format(
                    elapsed_time,
                    batch_num + 1,
                    batches_per_epoch,
                    losses.mean,
                    auc_score))
            

    def predict(self, data_loader):
        """Returns Chowder model predictions
        
           Args:
               data_loader: pytorch Data Loader to iterate over batches
        """
        # deactivates dropout
        self.set_predict_mode()
        preds = []
        start_time = time.time()
        
        for batch_num, X in enumerate(data_loader):
            # when passing in validation set X will be passed (X, y)
            if (type(X) == list) or (type(X) == tuple):
                X = X[0]
            
            # call .no_grad() to turn off backprop
            with torch.no_grad():
                y_pred = self.model(X)
            preds.append(y_pred.cpu().data.numpy())
        
        return np.concatenate(preds)
    
    
    def save(self, fn):
        self.model.save_model(fn)
        
    
    def load(self, fn):
        self.model.load_model(fn)