from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import os

class imageClassifierDataSet:
    """
    Goals of the MLDataSet class:
    (1) representing a set of images [providing paths] and corresponding labels
    (2) splitting the data set using the train/valid/test or k-folds paradygm
    (3) saving splited dataset into csv files
    """
    def __init__(self, relative_paths, labels, dataset_path=None,
                 data_directory="./data",target_size=(256,256),color_mode="rgb"):
        """
        :param relative_paths: list or array of strings. Relative paths of
        the images composing the data set
        :param labels: list or array of strings or integers. Labels of the images.
        :param dataset_path: string. Path to dataset saved in csv (used if given)
        :param data_directory: string. The path to the data/ directory
        :param target_size: tuple of integers (height,width). The dimension
        to which images found will be resized
        :param color_mode: string. The color system of images
        """
        self.color_mode=color_mode
        self.target_size=target_size
        if type(dataset_path)==type("") and os.path.isfile(dataset_path):
            self.load_dataset(dataset_path)
        else:
            assert type(relative_paths) == type([]) or \
                   type(relative_paths) == type(np.array([]))
            assert type(labels) == type([]) or type(labels) == type(np.array([]))
            if type(relative_paths)==type([]): relative_paths=np.array(relative_paths)
            if type(labels)==type([]): labels=np.array(labels)
            self.ndata = labels.shape[0]
            self.classes = list(np.unique(labels))
            self.labels = labels
            self.paths = [os.path.abspath(os.path.join(data_directory,pth))
                          for pth in relative_paths]

    def get_generator(self, indices, batch_size=32):
        """
        return: data generator with training images
        """
        dframe = DataFrame(data={"paths": self.paths[indices],
                                 "labels": self.labels[indices]})
        dgentr = ImageDataGenerator()
        return dgentr.flow_from_dataframe(dframe, x_col="paths", y_col="labels",
                                          target_size=self.target_size,
                                          color_mode=self.color_mode,
                                          shuffle=False, batch_size=batch_size,
                                          class_mode="categorical", classes=self.classes)

    def get_train_validation_test(self, test_size=None, validation_size=0., train_size=None,
                                  stratified=True, random_state=10, batch_size=32):
        """
        Split the dataset into a training, a validation and a test samples

        :param test_size : float, int or None (option, default=None).
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
        to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25.

        :param train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
        to include in the train split.
        If int, represents the absolute number of train samples.
        If None, the value is automatically set to the complement of the test size.

        :param random_state : int. The seed used by the random number generator;

        :param stratified: boolean, optional (default=True)
        If True, the sets are made by preserving the percentage of samples for each class.

        :param batch_size: integer. Size of batches of images

        :return: generators with training, validation and test images
        """
        indices = np.arange(self.ndata)
        rs_size = train_size+validation_size
        if stratified:
            tr_ind, ts_ind, tr_lab, ts_lab  = train_test_split(indices, self.labels,
                                                               test_size=test_size,
                                                               train_size=rs_size,
                                                               random_state=random_state, shuffle=True,
                                                               stratify=self.labels)
            if validation_size>0:
                rs_ind = tr_ind
                tr_ind, vl_ind, tr_lab, vl_lab  = train_test_split(rs_ind,
                                                                   self.labels[rs_ind],
                                                                   test_size=validation_size/rs_size,
                                                                   train_size=train_size/rs_size,
                                                                   random_state=random_state,
                                                                   shuffle=True,
                                                                   stratify=self.labels[rs_ind])

        else:
            tr_ind, ts_ind, tr_lab, ts_lab  = train_test_split(np.arange(self.ndata), self.labels,
                                                               test_size=test_size, train_size=rs_size,
                                                               random_state=random_state, shuffle=True)
            if validation_size>0:
                rs_ind = tr_ind
                tr_ind, vl_ind, tr_lab, vl_lab  = train_test_split(np.arange(rs_ind.size),
                                                                   self.labels[rs_ind],
                                                                   test_size=validation_size/rs_size,
                                                                   train_size=train_size/rs_size,
                                                                   random_state=random_state,
                                                                   shuffle=True)
        gen_train = self.get_generator(tr_ind,batch_size=batch_size)
        gen_test = self.get_generator(ts_ind,batch_size=batch_size)
        gen_valid = None
        if validation_size>0: gen_valid = self.get_generator(vl_ind,batch_size=batch_size)
        return gen_train, gen_valid, gen_test

    def get_KFolds(self, n_folds=5, validation_size=0,
                    stratified=True, random_state=10, batch_size=32):
        """
        Split the dataset into n_folds folds

        :param n_folds :  integer. Number of folds.

        :param validation_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion of the
        (n_folds-1) dataset dedicated to validation.
        If int, represents the absolute number of validation samples.

        :param random_state : int. The seed used by the random number generator;

        :param stratified: boolean, optional (default=True)
        If True, the sets are made by preserving the percentage of samples for each class.

        :param batch_size: integer. Size of batches of images

        :return: generators with training, validation and test images
        """
        indices = np.arange(self.ndata)
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        gen_folds = []
        for [rs_ind,ts_ind] in kf(indices,self.labels):
            if validation_size==0:
                gen_folds.append([self.get_generator(rs_ind,batch_size=batch_size),
                                  None, self.get_generator(ts_ind,batch_size=batch_size)])
            else:
                if stratified:
                    tr_ind, vl_ind, tr_lab, vl_lab  = train_test_split(rs_ind, self.labels[rs_ind],
                                                                       test_size=validation_size,
                                                                       train_size=None,
                                                                       random_state=random_state, shuffle=True,
                                                                       stratify=self.labels[rs_ind])
                else:
                    tr_ind, vl_ind, tr_lab, vl_lab  = train_test_split(rs_ind, self.labels[rs_ind],
                                                                       test_size=validation_size,
                                                                       train_size=None,
                                                                       random_state=random_state, shuffle=True)
                gen_folds.append([self.get_generator(tr_ind,batch_size=batch_size),
                                  self.get_generator(vl_ind,batch_size=batch_size),
                                  self.get_generator(ts_ind,batch_size=batch_size)])
        return gen_folds

    def save_dataset(self,filename,directory="./log"):
        """
        Save the dataset as a csv table
        :param filename:
        :param directory: path to the directory where csv file should be saved
        :return:
        """
        head = "# Total: %i images\n"%self.ndata
        for cls in self.classes:
            head += "# %s: %i images\n"%(cls,np.size(np.where(self.labels==cls)))
        #save in txt file
        if len(np.shape(self.labels))==1:
            data = np.transpose(np.array(self.paths,self.labels))
        else:
            data = np.concatenate([np.reshape(self.paths,(self.ndata,1)),
                                   self.labels],axis=1)
        np.savetxt(os.path.abspath(os.path.join(directory,filename)), data,
                   delimiter=',', fmt="%s", header=head)

    def load_dataset(self,path):
        """
        Load the training, validation and test sets from file
        :param path: string. Path to csv file
        :return:
        """
        tmp = np.loadtxt(path, delimiter=',', comments="#")
        self.labels = tmp[:,1:]
        self.paths = tmp[:,0]
        self.ndata = (self.labels).shape[0]
        self.classes = list(np.unique(self.labels))