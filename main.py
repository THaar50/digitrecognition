import pandas as pd
from keras import layers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def load_data(path: str):
    """Loads data from csv into a dataframe."""
    x_train = pd.read_csv(path)
    return x_train


def rescale_data(data):
    """Rescales numpy array to interval [0,1]."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def reshape_data(data):
    """Reshapes data to fit model structure. 42000 28x28 pixel images."""
    return np.expand_dims(rescale_data(data.reshape(data.shape[0], 28, 28)), -1)


def extract_transform_labels(data):
    """Extract and transform label column to categorical numpy array"""
    labels = data['label'].values
    return keras.utils.to_categorical(labels, len(np.unique(labels)))


def plot_first_n_instances(data, labels, n):
    """Plots first n images in data"""
    fig, axes = plt.subplots(1, n, figsize=(10, 3))
    for i in range(n):
        axes[i].set_title(str(labels[i]))
        axes[i].imshow(data[i], cmap='gray')
        axes[i].axis('off')
    plt.show()


def fit_model(x_train, filename):
    y_train = extract_transform_labels(x_train)
    num_classes = len(y_train[0])
    x_train = x_train.drop('label', axis=1)
    x_train = reshape_data(rescale_data(x_train.values))

    model = keras.Sequential(
        layers=[keras.Input(shape=x_train[0].shape),  # 28x28 pixel, one input channel (b/w)
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")],
        name='CNN'
    )
    model.summary()

    batch_size = 128
    epochs = 15

    print('Train neural net:')
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)

    score = model.evaluate(x_train, y_train, verbose=0, return_dict=True)
    print('Accuracy on test data: %.3f' % score['accuracy'])

    if filename:
        model.save(f"{filename}.h5")

    return model


def get_prediction(model, x_test, batch_size):
    """Returns predicted classes for given data."""
    prediction = model.predict(x_test, batch_size=batch_size)
    prediction_classes = np.argmax(prediction, axis=-1)
    return prediction_classes


def save_to_submission(class_predictions):
    """Saves predicted classes for test data to submission file."""
    submission = pd.read_csv('sample_submission.csv', index_col=0)
    submission['Label'] = class_predictions
    submission.to_csv('sample_submission_result.csv')


def main():
    use_pretrained = input('Use pretrained model (y/n)? ')
    if use_pretrained == 'y':
        filename = input('Please specify name of the file to load: ')
        try:
            model_fit = load_model(filepath=f"{filename}.h5")
        except OSError as e:
            print(f"Pretrained model file {filename}.h5 does not exist: {e}")
            return
    elif use_pretrained == 'n':
        filename = input('Please select a name for your model file. In case you do not wish to save your model just hit enter: ')
        model_fit = fit_model(x_train=load_data('data/train/train.csv'), filename=filename)
    else:
        print('Please select from [y/n]!')
        main()
        return
    x_test = reshape_data(data=load_data('data/test/test.csv').values)
    predictions = get_prediction(model=model_fit, x_test=x_test, batch_size=128)
    plot_first_n_instances(data=x_test, labels=predictions, n=50)
    save_to_submission(class_predictions=predictions)


if __name__ == '__main__':
    main()


