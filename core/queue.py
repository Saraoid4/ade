# TODO: deal with more then one element to enque and deque

class Queue:

    def __init__(self):
        self._queue = []
        self.size = 0

    def enqueue(self, item):
        if isinstance(item, list):
            self._queue.extend(item)
            self.size += len(item)
        else:
            self._queue.append(item)
            self.size += 1

    def dequeue(self, nb_elements):
        if self.has_next():
            first = self._queue[0: nb_elements]
            del self._queue[0: nb_elements]
            self.size = self.size - nb_elements if self.has_next() else 0
            return first
        else:
            raise IndexError('Empty queue, could not dequeue')

    def has_next(self):
        return self.size > 0
