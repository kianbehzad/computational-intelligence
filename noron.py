
class Noron:
    def __init__(self):
        self.numberOfInputs = 0
        self.weights = []

    def setWeight(self, weights:list[float]):
        self.numberOfInputs = len(weights)
        self.weights = weights