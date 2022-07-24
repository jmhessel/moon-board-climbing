import os

import json

class Model:
    def name(self):
        return "T5"

    def train(self, x_train, y_train):

        with open('train.json', 'w') as f:
            f.write(json.dumps({'x_train': list(x_train), 'y_train': list([int(x) for x in y_train])}))
        
        quit()
        num_classes = y_train.shape[1]
        num_climbs = x_train.shape[0]

        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(12, 2), dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(num_classes, activation="softmax"))
        self.model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=50, batch_size=num_climbs, verbose=0)

    def sample(self, x):
        y_pred = self.model.predict(x)
        bool_preds = [[1 if i == max(row) else 0 for i in row] for row in y_pred]
        return np.asarray(bool_preds)
