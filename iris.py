from os import path
from keras.engine.saving import model_from_json
from pandas import read_csv, get_dummies, DataFrame
from keras import models, layers, optimizers
from matplotlib.pyplot import figure, ylabel, plot, legend, show, xlabel
from numpy import reshape, argmax, ndarray

# These are the file names our model is
# going to be stored as
model_struct_f = "model.json"
weights_f = "weights.h5"

# If both of the required files exist, then we
# recreate our pre-trained model from them and
# ask for user input of a vector
if path.isfile(model_struct_f) and path.isfile(weights_f):
    print("Loading model struct and weights")
    with open(model_struct_f, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_f)

    # This lets us know that the model is ready
    print("The model is ready to make predictions!\nInput four floats delimited with spaces: (eg.: 6.9 3.1 4.9 1.5)\n")

    # Read four digits from stdin into an array
    # and reshape it to our model's input shape
    array = reshape(list(map(float, input().split())), (1, 4))

    # This will return an (1, 3) shape array with each column
    # representing how likely the input belongs to each class
    probabilities = model.predict(array)

    # Get the index of the highest value in our probabilities array
    # This will also be the index of the category or class name
    max_arg_i = argmax(probabilities)

    # If max_arg_i is an array it means there was more than one max
    # prob. values. In this case, just use the first one as the index
    if max_arg_i is ndarray:
        max_arg_i = max_arg_i[0]

    # These are just for pretty-printing the output in a human
    # readable format
    cat = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'][max_arg_i]
    confidence = probabilities[0, max_arg_i]

    print("I think it's an %s with %.3f%% confidence" % (cat, confidence))
else:
    # Load data from the internet
    data = read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

    # Shuffle data to avoid overfitting
    data = data.sample(frac=1).reset_index(drop=True)

    # Let's take a look at our data
    print(data.head(10))

    # the 0th, 1st, 2nd, 3th cols are the feature columns
    feature_cols = [0, 1, 2, 3]
    label_cols = [4]

    # replace missing values with the median of all values
    # in each feature column
    data[feature_cols].fillna(data[feature_cols].dropna().median(), inplace=True)

    # separate features and labels and OHE encode labels
    # We could use column slicing like this:
    #       x = data.values[:, :3]  # train data
    #       y = data.values[:, 4]   # labels
    # but since we have already named our feature columns, we can say:
    x = data[feature_cols].values

    # We could now say, well the labels are all the data cols except
    # the feature columns like this:
    #       y = data[data.columns.difference(feature_cols)].values
    # but it's much easier to just define the index of the label columns
    # y = data[label_cols] <--- omitted ---> to avoid code duplication
    y = get_dummies(data[label_cols], columns=label_cols).values

    # Now we'll divide our data set into train and test data
    div = int(x.shape[0] * 0.75)
    train_x, train_y = x[:div], y[:div]
    test_x, test_y = x[div:], y[div:]

    # Build the model
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(),
                  loss='categorical_crossentropy',  # use binary_crossentropy when dealing with only 2 classes
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_x, train_y, epochs=50, batch_size=15, verbose=1)

    # Evaluate on the test data
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print('Test loss %f\nTest accuracy %f%%' % (test_loss, test_acc*100))

    # Show a fancy graph of the training accuracy
    hist = DataFrame(history.history)
    hist['epoch'] = history.epoch
    figure()
    xlabel('Epoch')
    ylabel('Accuracy')
    plot(hist['epoch'], hist['accuracy'], label='Accuracy')
    legend()
    show()

    # Save the model struct and weights
    # so we don't have to re-train each run
    with open(model_struct_f, 'w') as f:
        f.write(model.to_json())
    model.save_weights(weights_f)
