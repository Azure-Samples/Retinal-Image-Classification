
# Copyright (C) Microsoft Corporation. All rights reserved.

# Licensed under the MIT License. See LICENSE in the project root for

# information.

from cvtk import Dataset
from cvtk import Splitter
from cvtk import DNNModel, TransferLearningModel
from cvtk import (ClassificationDataset, CNTKTLModel, Context, Splitter, StorageContext)
from cvtk.augmentation import augment_dataset
from cvtk.evaluation import ClassificationEvaluation
from cntk import softmax
import cvtk
import cntk
import os, sys, shutil, json
import glob
import pandas as pd
from imgaug import augmenters
import numpy as np
from IPython.display import display
from cvtk import Context


##### Load training dataset
dataset_train = ClassificationDataset.create_from_dir('CNV', 'C:\\OCT\\OCT2017\\OCT2017\\train')

##### Split training data
splitter = cvtk.Splitter(dataset_train)
trainSet, evalSet = splitter.split(0.80)

##### Model definition
class_map = {i:l.name for i, l in enumerate(dataset_train.labels)}
mymodel = CNTKTLModel(trainSet.labels, class_map=class_map,\
                      base_model_name='ResNet50_ImageNet_CNTK', output_path='C:\\CVTK\\OCT_output',\
                      image_dims=(3, 224, 224), num_conv_layers_freeze=0)

###### Start training
T = mymodel.train(trainSet, lr_per_mb=[0.01] * 7+ [0.001] * 7 + [0.0001],\
                   num_epochs=2, l2_reg_weight=0.005, mb_size=16)

###### Evaluate model on evaluation set
# Compute Accuracy, confusion matrix, and pr curve for evaluation set

ce = ClassificationEvaluation(mymodel, evalSet, minibatch_size=64)
acc = ce.compute_accuracy()
cm = ce.compute_confusion_matrix()
precisions, recalls, thresholds = ce.compute_precision_recall_curve()

##### Evaluate model on test set
testset_dir = 'C:\\OCT\\OCT2017\\OCT2017\\test'
dataset_test = ClassificationDataset.create_from_dir('CNV', testset_dir)

ce = ClassificationEvaluation(mymodel, dataset_test,\
                              minibatch_size=64, add_softmax=True)
acc = ce.compute_accuracy()
cm = ce.compute_confusion_matrix()
precisions, recalls, thresholds = ce.compute_precision_recall_curve()
