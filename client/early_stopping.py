class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  [EarlyStopping] Dihentikan.")
        return self.should_stop
