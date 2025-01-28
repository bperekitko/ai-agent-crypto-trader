from typing import TypeVar, Generic, List

DEQUEUE_ELEMENT_TYPE = TypeVar('DEQUEUE_ELEMENT_TYPE')


class Dequeue(Generic[DEQUEUE_ELEMENT_TYPE]):

    def __init__(self, size):
        self.size = size
        self.elements: List[DEQUEUE_ELEMENT_TYPE] = []

    def push(self, element: DEQUEUE_ELEMENT_TYPE):
        self.elements.append(element)
        if len(self.elements) > self.size:
            del self.elements[0]

    def current_size(self):
        return len(self.elements)

    def get_all(self):
        return self.elements.copy()
