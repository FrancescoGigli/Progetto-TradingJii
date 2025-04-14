import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from keras.losses import Loss
from keras import backend as K
from keras_tuner import HyperModel
from sklearn.ensemble import RandomForestClassifier
import logging
from config import TIME_STEPS, EXPECTED_COLUMNS

class FocalLoss(Loss):
    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_true = K.cast(y_true, dtype=tf.float32)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)
        weight = (self.alpha * y_true * K.pow((1.0 - y_pred), self.gamma) +
                  (1.0 - self.alpha) * (1.0 - y_true) * K.pow(y_pred, self.gamma))
        loss = weight * cross_entropy
        return K.mean(loss)

def create_lstm_model(input_shape):
    """
    Crea un modello LSTM semplice per il trading.
    L'input shape viene impostato in base a TIME_STEPS e al numero di feature in EXPECTED_COLUMNS.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
    return model

class LSTMHyperModel(HyperModel):
    """
    Modello iperparametrico per la ricerca automatica (tuning) del modello LSTM.
    """
    def build(self, hp):
        num_features = len(EXPECTED_COLUMNS)
        input_shape = (TIME_STEPS, num_features)
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Bidirectional(LSTM(units=hp.Int('units1', 32, 256, step=32), return_sequences=True)))
        model.add(Dropout(rate=hp.Float('dropout1', 0.1, 0.5, step=0.1)))
        model.add(Bidirectional(LSTM(units=hp.Int('units2', 32, 256, step=32))))
        model.add(Dropout(rate=hp.Float('dropout2', 0.1, 0.5, step=0.1)))
        model.add(Dense(units=hp.Int('dense_units', 16, 128, step=16), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
        logging.info("Modello LSTM costruito con iperparametri: " + str(hp.values))
        return model

def create_rf_model():
    """
    Crea e restituisce un modello Random Forest configurato per il trading.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Utilizza tutti i core della CPU
    )
    return rf