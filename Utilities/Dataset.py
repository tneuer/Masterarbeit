#!/usr/bin/env python
"""
####################################################################################
    # -*- coding: utf-8 -*-
    # Author  : Thomas Neuer (tneuer)
    # Creation Date : 2019-11-25 15:41:09
    # Description :
####################################################################################
"""
import numpy as np

class Dataset():
    def __init__(self, x, y=None, shuffle=True, seed=42):
        if y is not None:
            assert len(x) == len(y), "x and y have to have same length, given: x={}, y={}.".format(len(x), len(y))
        if seed is not None:
            self.set_seed(seed)

        self._xdata = x
        self._ydata = y
        self._n = len(x)
        self._indices = np.arange(self._n)
        self._batch_nr = 0
        self._shuffle = shuffle
        self._offset = 0

        if self._shuffle:
            self.shuffle_indices()

        if self._ydata is None:
            self.get_next_batch = self._get_next_batch_x
            self.sample = self._sample_x
        else:
            self.get_next_batch = self._get_next_batch_xy
            self.sample = self._sample_xy


    def _get_next_batch_xy(self, batch_size):
        next_batch_indices = self._get_next_batch_indices(batch_size)
        next_batch_x = self._xdata[next_batch_indices]
        next_batch_y = self._ydata[next_batch_indices]

        return next_batch_x, next_batch_y


    def _get_next_batch_x(self, batch_size):
        next_batch_indices = self._get_next_batch_indices(batch_size)
        next_batch = self._xdata[next_batch_indices]

        return next_batch


    def _get_next_batch_indices(self, batch_size):
        if self._offset+(self._batch_nr+1)*batch_size <= self._n:
            next_batch_indices = self._indices[(self._offset+self._batch_nr*batch_size):(self._offset+(self._batch_nr+1)*batch_size)]
        else:
            next_batch_indices = self._indices[self._offset+self._batch_nr*batch_size:]
            if self._shuffle:
                self.shuffle_indices()
            self._offset = batch_size - len(next_batch_indices)
            self._batch_nr = 0
            next_batch_indices = np.concatenate((next_batch_indices, self._indices[:self._offset]), axis=None)

        self._batch_nr += 1
        return next_batch_indices


    def shuffle_indices(self):
        np.random.shuffle(self._indices)

    def __len__(self):
        return(self._n)

    def get_xdata(self):
        return self._xdata

    def get_ydata(self):
        return self._ydata

    def _sample_x(self, n, keep=True):
        random_indices = self.get_random_indices(n=n)
        sample_x = self._xdata[random_indices]
        if not keep:
            self._xdata = np.delete(self._xdata, random_indices, axis=0)
            self.reset_indices()
        return sample_x

    def _sample_xy(self, n, keep=True):
        random_indices = self.get_random_indices(n=n)
        sample_x, sample_y = self._xdata[random_indices], self._ydata[random_indices]
        if not keep:
            self._xdata = np.delete(self._xdata, random_indices, axis=0)
            self._ydata = np.delete(self._ydata, random_indices, axis=0)
            self.reset_indices()
        return sample_x, sample_y

    def get_random_indices(self, n):
        return np.random.choice(self._indices, size=n, replace=False)

    def set_seed(self, seed):
        np.random.seed(seed)

    def reset_indices(self):
        self._n = len(self._xdata)
        self._indices = np.arange(self._n)
        self.shuffle_indices()

