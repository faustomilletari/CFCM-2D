from multiprocessing import Process, Manager

import numpy as np


class BatchIterator(object):
    def __init__(
            self,
            epoch_transforms=[],
            iteration_transforms=[],
            batch_size=32,
            iteration_keys=None,
    ):
        self.epoch_transforms = epoch_transforms
        self.iteration_transforms = iteration_transforms
        self.batch_size = batch_size
        self.iteration_keys = iteration_keys
        self.data = None

    def __call__(self, data):
        self.data = data

        if self.iteration_keys is None:
            self.iteration_keys = self.data.keys()

    def __iter__(self):

        for transform in self.epoch_transforms:
            self.data = transform(self.data)

        data_length = self.data[self.iteration_keys[0]].shape[0]

        num_batches = np.floor(data_length / self.batch_size)

        if (data_length % self.batch_size) != 0:
            num_batches += 1

        for i in range(int(num_batches)):
            batch = {}
            batch_start = i * self.batch_size
            batch_stop = np.min([data_length + 1, batch_start + self.batch_size])

            for key in self.iteration_keys:
                batch[key] = self.data[key][batch_start:batch_stop]

            for transform in self.iteration_transforms:
                batch = transform(batch)

            yield batch


class ParallelBatchIterator(BatchIterator):
    def __init__(
            self,
            shuffle=False,
            iteration_transforms=[],
            batch_size=32,
            iteration_keys=None,
            number_threads=16,
            max_batch_queue_size=128,
    ):
        super(ParallelBatchIterator, self).__init__(
            epoch_transforms=[],
            iteration_transforms=iteration_transforms,
            batch_size=batch_size,
            iteration_keys=iteration_keys
        )
        self.shuffle = shuffle
        self.number_threads = number_threads
        self.max_batch_queue_size = max_batch_queue_size

        self.batch_start_stop = Manager().Queue(maxsize=1000)
        self.batch_queue = Manager().Queue(maxsize=self.max_batch_queue_size)
        self.should_run = True

        self.p_list = []

    def __call__(self, data):
        super(ParallelBatchIterator, self).__call__(data)

        data_length = self.data[self.iteration_keys[0]].shape[0]

        num_batches = np.floor(data_length / self.batch_size)

        if (data_length % self.batch_size) != 0:
            num_batches += 1

        def create_batches():
            # done by just one process. (sequential)
            while self.should_run:
                if self.shuffle:
                    new_indices = np.random.permutation(data_length)
                else:
                    new_indices = range(data_length)

                for i in range(int(num_batches)):
                    batch_start = i * self.batch_size
                    batch_stop = np.min([data_length + 1, batch_start + self.batch_size])
                    indices = new_indices[batch_start:batch_stop]
                    self.batch_start_stop.put(indices)

        def load_batches():
            np.random.seed()
            while self.should_run:
                batch = {}

                indices = self.batch_start_stop.get(block=True)

                for key in self.iteration_keys:
                    batch[key] = self.data[key][indices]

                for transform in self.iteration_transforms:
                    batch = transform(batch)

                self.batch_queue.put(batch)

        for i in range(self.number_threads):
            p = Process(target=load_batches, args=())
            p.start()
            self.p_list.append(p)

        pb = Process(target=create_batches, args=())
        pb.start()
        self.p_list.append(pb)

    def __iter__(self):
        data_length = self.data[self.iteration_keys[0]].shape[0]

        num_batches = np.floor(data_length / self.batch_size)

        if (data_length % self.batch_size) != 0:
            num_batches += 1

        for i in range(int(num_batches)):
            batch = self.batch_queue.get(block=True)
            yield batch

    def __del__(self):
        self.should_run = False

        self.batch_queue.clear()  # in case process is waiting to put something in a full queue

        for p in self.p_list[:-1]:
            p.join()

        self.batch_start_stop.clear()  # in case process is waiting to put something in a full queue

        self.p_list[-1].join()
