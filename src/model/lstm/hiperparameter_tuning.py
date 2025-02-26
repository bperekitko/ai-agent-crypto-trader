import numpy as np
import pandas as pd
from keras.src.models import Sequential
from keras.src.layers import Input, LSTM, Dropout, BatchNormalization, Dense
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.utils.class_weight import compute_class_weight

from model.lstm.binary_lstm import LongTradeLstm


def build_model(hp):
    model = Sequential()

    num_lstm_layers = hp.Int("num_lstm_layers", min_value=1, max_value=2, step=1)
    lstm_units = hp.Int("lstm_units_1", min_value=32, max_value=256, step=32)

    model.add(Input(shape=(24, len(LongTradeLstm().features))))
    model.add(LSTM(units=lstm_units, return_sequences=(num_lstm_layers > 1)))

    dropout_rate = hp.Float("dropout_1", min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    if num_lstm_layers > 1:
        lstm_units_2 = hp.Int("lstm_units_2", min_value=32, max_value=128, step=32)
        model.add(LSTM(units=lstm_units_2, return_sequences=False))
        dropout_rate_2 = hp.Float("dropout_2", min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_rate_2))
        model.add(BatchNormalization())

    num_dense_layers = hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)

    dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)
    model.add(Dense(units=dense_units, activation="relu"))
    dropout_dense = hp.Float("dropout_dense1", min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_dense))

    if num_dense_layers > 1:
        dense_units2 = hp.Int("dense_units2", min_value=16, max_value=128, step=16)
        model.add(Dense(units=dense_units2, activation="relu"))
        dropout_dense = hp.Float("dropout_dense2", min_value=0.1, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_dense))

    model.add(Dense(1, activation="sigmoid"))

    learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["precision"])
    return model


def tune_lstm(df: pd.DataFrame):
    model = LongTradeLstm()
    split_index = int(len(df) * 0.85)
    train_df = df.iloc[:split_index].copy()
    val_df = df.iloc[split_index:].copy()

    train_data = model.prepare_data(train_df)
    val_data = model.prepare_data(val_df)

    x_train, y_train = model.to_sequences(train_data)
    x_val, y_val = model.to_sequences(val_data)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_for_model = dict(zip(np.unique(y_train), class_weights))

    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=100,
        executions_per_trial=1,
        directory="keras_tuner_dir",
        project_name="lstm_model_optimization",
        overwrite=True
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, mode="min")
    tuner.search(
        x_train, y_train,
        epochs=100,
        batch_size=8,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
        class_weight=class_weights_for_model
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Najlepsze hiperparametry:")
    print(best_hp.values)
