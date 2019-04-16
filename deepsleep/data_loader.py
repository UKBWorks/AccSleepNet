import os

import numpy as np
import pandas as pd
from deepsleep.sleep_stage import print_n_samples_each_class
from deepsleep.utils import get_balance_class_oversample
from random import shuffle,seed
import re


class NonSeqMESADataLoader(object):
    def __init__(self, data_path='', save_filepath='', n_folds=0, fold_idx=0, random_seed=0, percentage_test=0.2,
                 use_npz=True):
        self.data_path = data_path
        self.save_filepath= save_filepath
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.longest_seq = 2608
        self.random_se = random_seed
        self.percent_test = percentage_test
        #self.dimension = dimension
        seed(self.random_se)
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.test_list = []
        self.train_list = []
        #self.nb_classes = nb_classes
        if use_npz:
            self._load_npz()
        else:
            self.csv_files = self._get_csv_files()
            self.padding_pond = pd.DataFrame()
            self._build_padding_pond()
            self._padding_train_framewise()

    def _load_npz(self):
        allfiles = os.listdir(self.save_filepath)
        npz_files = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npz_files.append(os.path.join(self.save_filepath, f))
        npz_files.sort()
        Xtrain_tmp = []
        Ytrain_tmp = []
        for _, npz in enumerate(npz_files):
            with np.load(self.save_filepath) as f:
                Xtrain_tmp = Xtrain_tmp + f["new_train_data"]
                Ytrain_tmp = Ytrain_tmp + f["new_train_label"]
        split = int(self.percent_test * len(Xtrain_tmp))
        self.X_train = np.vstack(Xtrain_tmp[:split])
        self.X_train = np.nan_to_num(self.X_train)  # check nan

        self.Y_train = np.hstack(Ytrain_tmp[:split])
        self.Y_train = np.where(self.Y_train > 1, 1, 0)

        self.X_test = np.vstack(Xtrain_tmp[split:])
        self.X_test = np.nan_to_num(self.X_test) # check nan

        self.Y_test = np.hstack(Ytrain_tmp[split:])
        self.Y_test = np.where(self.Y_test > 1, 1, 0)

    def _get_csv_files(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_path)
        csv_files = []
        for idx, f in enumerate(allfiles):
            if ".csv" in f:
                csv_files.append(os.path.join(self.data_path, f))
        csv_files.sort()
        if self.longest_seq == 0:
            for _, file in enumerate(csv_files):
                num_lines = sum(1 for line in open(file))
                self.longest_seq = num_lines if self.longest_seq < num_lines else self.longest_seq
        return csv_files

    # def _new_split_train_test(self):
    #     total_length = len(self.csv_files)
    #     num_train = int(np.round(total_length * (1-0.2), 0))
    #     shuffle(self.csv_files)
    #     self.test_list = self.csv_files[0:num_train]  # ! TODO change it to num_test
    #     self.train_list = self.csv_files[num_train:total_length]  # ! TODO change it to total_length

    def _build_padding_pond(self):
        pond_file_list = self.train_list[:200]
        if os.path.exists(os.path.join(self.save_filepath, "ponding_dataframe.csv")):
            self.padding_pond = pd.read_csv(os.path.join(self.save_filepath, "ponding_dataframe.csv"))
            return
        if self.padding_pond.shape[0] == 0:
            self.padding_pond = pd.read_csv(pond_file_list[0])
            # self.padding_pond = self._preprocess_pd(self.padding_pond)
            self.padding_pond = self.padding_pond[self.padding_pond['stage'] == 0.0]
        for _, file in enumerate(pond_file_list):
            _tmp_pd = pd.read_csv(file)
            _tmp_pd = _tmp_pd[_tmp_pd['two_stage'] == 0.0]
            self.padding_pond = pd.concat([self.padding_pond, _tmp_pd], ignore_index=True, axis=0)
        print("Build activity pond is completed")
        self.padding_pond.to_csv(os.path.join(self.save_filepath, "ponding_dataframe.csv"))


    def _gen_more_data(self, data_set, num_gen):
        gen_array = []
        if len(data_set) > 0 and num_gen > 0:
            idx = np.random.randint(len(data_set), size=(num_gen))
            gen_array = data_set[idx]
            gen_array = np.transpose(gen_array)
        return gen_array

    def _sliding_windows_transformer(self, train_data, wake_data_pool_avg, window_size=5):
        '''

        :param npz_files:
        :param epoch_length:
        :param window_size: minutes based
        :return:
        '''

        half_window_size = window_size // 2
        dataset_len = len(train_data)
        new_dataset = []
        for i in np.arange(len(train_data)):
            start_pad_idx = i - half_window_size
            end_pad_idx = i + half_window_size
            padded_data_entry = np.array([])
            # if the start index smaller than zero then padding with weakful data
            if start_pad_idx < 0:
                # only feed the first 360 records
                pad_start_data = self._gen_more_data(wake_data_pool_avg, np.abs(start_pad_idx))
                pad_start_data = np.append(pad_start_data, train_data[:i])
            else:
                pad_start_data = train_data[start_pad_idx:i]
            # if the index go beyond the total length of dataset
            if end_pad_idx >= dataset_len:
                pad_end_data = train_data[i:]
                tmp_entry = self._gen_more_data(wake_data_pool_avg, np.abs(half_window_size - (dataset_len - i) + 1))
                pad_end_data = np.append(pad_end_data, tmp_entry)
            # if the index value still below the total length
            else:
                pad_end_data = train_data[i:end_pad_idx + 1]

            padded_data_entry = np.hstack((padded_data_entry, pad_start_data))
            padded_data_entry = np.hstack((padded_data_entry, pad_end_data))
            padded_data_entry = padded_data_entry.reshape(padded_data_entry.shape[0], 1).transpose()
            if padded_data_entry.shape[1] != window_size :
                t1 = 1;
            new_dataset.append(padded_data_entry)
        new_dataset = np.vstack(new_dataset)
        new_dataset = np.asarray(new_dataset).astype(np.float32)
        new_dataset = np.reshape(new_dataset, new_dataset.shape + (1,))
        return new_dataset

    def _padding_train_framewise(self):
        XData = []
        YData = []
        for _, file in enumerate(self.csv_files):
            _tmp_pd = pd.read_csv(file)
            # append awake data to make all sequence the same size
            activity_data = _tmp_pd['activity'].tolist()
            activity_data_pool = np.asarray(self.padding_pond['activity'].tolist())
            tmp_train = self._sliding_windows_transformer(activity_data, activity_data_pool, window_size=721)

            print("After append, the size is: {}".format(tmp_train.shape))
            list_size_chk = np.array(_tmp_pd[['seconds', 'activity']].values.tolist())
            if len(list_size_chk.shape) < 2:
                print("File {f_name} doesn't meet dimension requirement, it's size is {wrong_dim}"
                      .format(f_name=file, wrong_dim=list_size_chk.shape))
            else:
                _tmp_pd = _tmp_pd[
                    ['HR', 'HR_max', 'HR_min', 'HR_std', 'RR Intervals', 'activity', 'seconds', 'stage','two_stage']
                ]

            XData.append(tmp_train)
            YData.append(_tmp_pd['two_stage'].values.tolist())
        split = int((1-self.percent_test) * len(XData))
        self.X_train = np.vstack(XData[:split])
        self.X_train = np.nan_to_num(self.X_train)  # check nan

        self.Y_train = np.hstack(YData[:split])
        self.Y_train = np.where(self.Y_train > 1, 1, 0)

        self.X_test = np.vstack(XData[split:])
        self.X_test = np.nan_to_num(self.X_test) # check nan

        self.Y_test = np.hstack(YData[split:])
        self.Y_test = np.where(self.Y_test > 1, 1, 0)
        # default time step = 2608, dimension = 7
        #self.X_train = np.reshape(self.X_train, (-1, self.longest_seq, self.dimension))

        #self.Y_train = np.expand_dims(self.Y_train, 1)
        #self.Y_train = np.reshape(self.Y_train, (-1, self.longest_seq, 1))
        #self.Y_train = np_utils.to_categorical(self.Y_train, self.nb_classes)



    def load_train_data(self):
        self.X_train = np.squeeze(self.X_train)
        self.X_test = np.squeeze(self.X_test)
        self.X_train = self.X_train[:, :, np.newaxis, np.newaxis]
        self.X_test = self.X_test[:, :, np.newaxis, np.newaxis]
        return self.X_train, self.Y_train, self.X_test, self.Y_test


class NonSeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            if ("x" in f.keys()):
                data = f["x"]
                labels = f["y"]
                sampling_rate = f["fs"]
            else:
                data = f["svm"]
                labels = f["annotation"]
                sampling_rate = f["fs"]

        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print ("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            if len(tmp_labels.shape) > 1:
                tmp_labels=np.squeeze(tmp_labels)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def _load_cv_data(self, list_files):
        """Load training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)

        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print(" ")

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_train, label_train, data_val, label_val

    def load_train_data(self, binary_sleep =True):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                #pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
                # the purpose of this line code is to specify which file should be considered as the validation
                # dataset
                pattern = re.compile("[a-zA-Z0-9. -_]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if "VALIDATION" in f:
                subject_files.append(os.path.join(self.data_dir, f))

        if len(subject_files) == 0:
            for idx, f in enumerate(allfiles):
                if self.fold_idx < 10:
                    pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(self.fold_idx))
                else:
                    pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(self.fold_idx))
                if "VALIDATION" in f:
                    subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(npz_files=subject_files)
        print(" ")

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)
        if binary_sleep:
            label_train[label_train > 1] = 1
            label_val[label_val > 1] = 1
        print("Training set: {}, {}".format(data_train.shape, label_train.shape))
        print_n_samples_each_class(label_train, binary=binary_sleep)
        print(" ")
        print("Validation set: {}, {}".format(data_val.shape, label_val.shape))
        print_n_samples_each_class(label_val, binary=binary_sleep)
        print(" ")

        # Use balanced-class, oversample training set
        x_train, y_train = get_balance_class_oversample(
            x=data_train, y=label_train
        )
        print("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        ))
        print_n_samples_each_class(y_train, binary=binary_sleep)
        print(" ")

        return x_train, y_train, data_val, label_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if "VALIDATION" in f:
                subject_files.append(os.path.join(self.data_dir, f))
        subject_files.sort()

        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))

        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(subject_files)

        # Reshape the data to match the input of the model
        data_val = np.squeeze(data_val)
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_val, label_val


class SeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["svm"]
            labels = f["annotation"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _load_cv_data(self, list_files):
        """Load sequence training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print(" ")


        return data_train, label_train, data_val, label_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        # Files for validation sets
        val_files = np.array_split(npzfiles, self.n_folds)
        val_files = val_files[self.fold_idx]

        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))

        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files)

        return data_val, label_val

    def load_train_data(self, n_files=None, binary_sleep = True):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        subject_files = []
        for idx, f in enumerate(allfiles):
            if self.fold_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
            if "VALIDATION" in f:
                subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(subject_files)
        print(" ")
        if binary_sleep:
            for y in label_train:
                y[y > 0] = 1
            for y in label_val:
                y[y > 0] = 1
        print("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0
        for d in data_train:
            print(d.shape)
            n_train_examples += d.shape[0]
        print("Number of examples = {}".format(n_train_examples))
        print_n_samples_each_class(np.hstack(label_train), binary=binary_sleep)
        print(" ")
        print("Validation set: n_subjects={}".format(len(data_val)))
        n_valid_examples = 0
        for d in data_val:
            print(d.shape)
            n_valid_examples += d.shape[0]
        print ("Number of examples = {}".format(n_valid_examples))
        print_n_samples_each_class(np.hstack(label_val), binary=binary_sleep)
        print (" ")

        return data_train, label_train, data_val, label_val

    @staticmethod
    def load_subject_data(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(data_dir)
        subject_files = []
        for idx, f in enumerate(allfiles):
            if subject_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(subject_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(subject_idx))
            if "VALIDATION" in f:
                subject_files.append(os.path.join(data_dir, f))

        # Files for validation sets
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_npz_file(npz_file):
            """Load data and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["svm"]
                labels = f["annotation"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and labels from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                print ("Loading {} ...".format(npz_f))
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data)
                tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels

        print("Load data from: {}".format(subject_files))
        data, labels = load_npz_list_files(subject_files)

        return data, labels
