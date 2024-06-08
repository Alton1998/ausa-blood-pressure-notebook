import numpy as np
import tensorflowjs as tfjs
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Multiply,
    Input,
    BatchNormalization,
    Activation,
    LSTM,
)
from tensorflow.keras.callbacks import EarlyStopping

def generate_fake_data(number_of_patients=365):
    X_context = []
    X_time_series = []
    y = []
    for _ in range(0, number_of_patients):
        age = np.random.randint(0, 101)
        sex = np.random.randint(0, 2)
        bmi = np.random.randint(30, 40)
        systolic_bp = np.random.randint(80, 120, size=14)
        diastolic_bp = np.random.randint(70, 80, size=14)
        future_systolic_bp = np.random.randint(80, 120, size=7)
        future_diastolic_bp = np.random.randint(70, 80, size=7)
        result = np.concatenate((future_systolic_bp, future_diastolic_bp), axis=0)
        X_context.append([age, sex, bmi])
        X_time_series.append([systolic_bp, diastolic_bp])
        y.append([result])
    return (
        np.array(X_context, dtype=float),
        np.array(X_time_series, dtype=float),
        np.array(y, dtype=float),
    )
X_context, X_time_series, y = generate_fake_data()
X_time_series = X_time_series.reshape(365, 14, 2)
y = y.reshape(365, 14, 1)
input_context_layer = Input((3,))
context_layer = Dense(3, activation="leaky_relu")(input_context_layer)
batch_norm_layer = BatchNormalization()(context_layer)
sigmoid_layer = Activation(activation="sigmoid")(batch_norm_layer)
input_time_series_layer = Input(
    (
        14,
        2,
    )
)
bp_sequence_layer = LSTM(3, activation="tanh")(input_time_series_layer)
multiply_layer = Multiply()([sigmoid_layer, bp_sequence_layer])
final_layer = Dense(14)(multiply_layer)
model = Model(
    inputs=[input_context_layer, input_time_series_layer], outputs=final_layer
)
model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=5)
model.fit(
    x=[X_context, X_time_series],
    y=y,
    batch_size=32,
    epochs=1000,
    validation_split=0.2,
    callbacks=[early_stop],
)
model.save("bp.keras")
tfjs.converters.save_keras_model(model, "./tensorflow_js")

