class Variable(object):
    """Stores the value and the gradient
    """

    def __init__(self):
        self.value = None
        self.grad = None
