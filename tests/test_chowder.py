import unittest

import torch
import torch.nn as nn
from chowder.model import ChowderArchitecture
from chowder.maxmin_pooling import MaxMinPooling


class TestChowderModel(unittest.TestCase):

    
    def test_init(self):
        arch = ChowderArchitecture(MaxMinPooling(R=3))

        self.assertEqual(arch.R, 3)
        self.assertEqual(len(list(arch.children())), 4)


    def test_forward(self):
        # test input with the correct number of dimensions
        # the input must be >= P * R

        model = ChowderArchitecture(MaxMinPooling(R=2))
        x = torch.randn(1, 1, 2048*1)

        with self.assertRaises(RuntimeError):
            model.forward(x)

        x = torch.randn(1, 1, 2048*4)
        try:
            model.forward(x)
        except RuntimeError:
            self.fail('Chowder model is not accepting correct input tensors')

        # propogate the network forward and back twice and validate if the
        # loss and predictions have changed since the first & second prediction
        x = torch.randn(3, 1, 2048*40)
        y = torch.FloatTensor([[0], [1], [0]])

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        optimizer.zero_grad()

        losses = []

        pred_1 = model.forward(x)
        loss = criterion(pred_1, y)
        losses.append(loss.item())

        # test if the loss is returning NaN
        try: 
            assert not torch.isnan(loss.data).byte().any()
        except:
            self.fail('Loss returned NaN!')

        loss.backward()
        optimizer.step()


        pred_2 = model(x)
        loss = criterion(pred_2, y)
        losses.append(loss.item())

        equal_pred = torch.equal(pred_1, pred_2)
        
        self.assertFalse(equal_pred)
        self.assertNotEqual(losses[0], losses[1])


    def test_backward(self):
        # Need to add this!
        pass



class TestMaxMinPooling(unittest.TestCase):

    def test_init(self):
        pooling = MaxMinPooling(R=5)
        self.assertEqual(pooling.R, 5)


    def test_forward(self):
        pooling = MaxMinPooling(R=2)

        x = torch.tensor([[[1, -1., 3, 0, 2]]])
        expected_output = torch.FloatTensor([[3, 2, 0, -1]])
        output = pooling.forward(x)

        equal_bool = torch.equal(output, expected_output)

        self.assertTrue(equal_bool)