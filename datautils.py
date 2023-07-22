import os.path

import torch
from PIL import Image
import PIL
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import random
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler


from NEWLoc import *
from labels_create import *


class ZSLDataset(Dataset):
    def __init__(self, dset, n_train, n_test, train=True, gzsl=False, synthetic=False, syn_dataset=None):
        '''
        Base class for all datasets
        Args:
            dset        : Name of dataset - 1 among [sun, cub, awa1, awa2]
            n_train     : Number of train classes
            n_test      : Number of test classes
            train       : Boolean indicating whether train/test
            gzsl        : Boolean for Generalized ZSL
            synthetic   : Boolean indicating whether dataset is for synthetic examples
            syn_dataset : A list consisting of 3-tuple (z, _, y) used for sampling
                          only when synthetic flag is True
        '''
        super(ZSLDataset, self).__init__()
        self.dset = dset
        self.n_train = n_train
        self.n_test = n_test
        self.train = train
        self.gzsl = gzsl
        self.synthetic = synthetic
        self.tranform_fn = self.tranform()
        self.dset =dset
        res101_data = scio.loadmat('./datasets_CWRU_0123/image_files_0123.mat')
        self.features = res101_data['image_files']
        self.root_dir = './datasets_CWRU_0123/images_0123'
        self.labels, self.valueall = LabelCreate(self.root_dir)
        self.labels =self.labels.reshape(-1)

        self.attribute_dict = scio.loadmat('./datasets_CWRU_0123/att_CWRU_03366.mat')
        self.attributes = self.attribute_dict['att'].T
        self.class_names_file = './datasets_CWRU_0123/classes_0123.txt'


        # test class names
        with open('./datasets_CWRU_0123/testclasses_0123.txt') as fp:
            self.test_class_names = [i.strip() for i in fp.readlines()]
        if self.synthetic:
            assert syn_dataset is not None
            self.syn_dataset = syn_dataset

        else:
            self.dataset = self.create_orig_dataset()
            if self.train:
                self.gzsl_dataset = self.create_gzsl_dataset()

    def normalize(self, matrix):
        scaler = MinMaxScaler()
        return scaler.fit_transform(matrix)

    def get_classmap(self):
        '''
        Creates a mapping between serial number of a class
        in provided dataset and the indices used for classification.
        Returns:
            2 dicts, 1 each for train and test classes
        '''
        with open(self.class_names_file) as fp:
            all_classes = fp.readlines()

        test_count = 0
        train_count = 0
        #val_count = 0

        train_classmap = dict()
        test_classmap = dict()
        #val_classmap = dict()
        for line in all_classes:
            idx, name = [i.strip() for i in line.split(' ')]

            # if name in self.val_class_names:
            #     val_classmap[int(idx)] = val_count
            #     val_count += 1

            if name in self.test_class_names:
                if self.gzsl:
                    # train classes are also included in test time
                    test_classmap[int(idx)] = self.n_train + test_count#
                else:
                    test_classmap[int(idx)] = test_count
                test_count += 1
            else:
                train_classmap[int(idx)] = train_count
                train_count += 1

        return train_classmap, test_classmap#, val_classmap  #{"1":0}，{“2”：0,“3”：1.....：0，“10”：9}

    def create_gzsl_dataset(self, n_samples=200):#TODO 200
        '''
        Create an auxillary dataset to be used during training final
        classifier on seen classes
        '''
        dataset = []
        for key, features in self.gzsl_map.items():
            if len(features) < n_samples:
                aug_features = [random.choice(features) for _ in range(n_samples)]
            else:
                aug_features = random.sample(features, n_samples)

            dataset.extend([(f, -1, key) for f in aug_features])
        return dataset

    def create_orig_dataset(self):
        '''
        Returns list of 3-tuple: (feature, label_in_dataset, label_for_classification)
        '''
        self.train_classmap, self.test_classmap = self.get_classmap()
        #TODO linadd
        print(f'n_test:{self.n_test}')

        trainval_loc, test_unseen_loc, test_seen_loc = newloc(self.valueall)

        if self.train:
            #labels = self.attribute_dict['trainval_loc'].reshape(-1)
            labels = trainval_loc.reshape(-1)#
            classmap = self.train_classmap
            self.gzsl_map = dict()
        else:
            #labels = self.attribute_dict['test_unseen_loc'].reshape(-1)
            labels = test_unseen_loc.reshape(-1)#TODO
            if self.gzsl:
                #labels = np.concatenate((labels, self.attribute_dict['test_seen_loc'].reshape(-1)))
                #test_seen_loc = test_seen_loc.reshape(test_seen_loc.shape[0],1)
                labels = np.concatenate((labels, test_seen_loc)).ravel()
                classmap = {**self.train_classmap, **self.test_classmap}
            else:
                classmap = self.test_classmap


        dataset = []
        #times =0
        for l in labels:
            idx = self.labels[l - 1]
            # if idx == 2:
            #
            #     print(f"l:{l}")
            #     print(f'idex:{idx}')
            #     times += 1
                #continue
            dataset.append((self.features[l - 1], idx, classmap[idx]))

            if self.train:
                # create a map bw class label and features
                try:
                    self.gzsl_map[classmap[idx]].append(self.features[l - 1])
                except Exception as e:
                    self.gzsl_map[classmap[idx]] = [self.features[l - 1]]
        #print(f'times:{times}')

        return dataset

    def __getitem__(self, index):
        if self.synthetic:
            # choose an example from synthetic dataset
            img_feature, orig_label, label_idx = self.syn_dataset[index]
            # print(img_feature,orig_label, label_idx)
        else:
            # choose an example from original dataset
            img_feature, orig_label, label_idx = self.dataset[index]

            img_feature = Image.open(img_feature[0][0])
            img_feature = self.tranform_fn(img_feature)
            ###======end============
        label_attr = self.attributes[orig_label - 1]
        return img_feature, label_attr, label_idx

    def __len__(self):
        if self.synthetic:
            return len(self.syn_dataset)
        else:
            return len(self.dataset)

    def tranform(self):
        t = []
        t.append( transforms.Resize(224, interpolation=PIL.Image.BICUBIC))  # to maintain same ratio w.r.t. 224 images
        t.append(transforms.ToTensor())  #
        t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        return transforms.Compose(t)


