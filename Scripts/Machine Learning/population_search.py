#pip install parameter-sherpa
#https://parameter-sherpa.readthedocs.io/en/latest/algorithms/keras_mnist_mlp_population_based_training.html

import os
import keras
import sherpa
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# 0. IMPORT THE DATASET
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
# 1. SETUP THE MODEL
def TheModel(nodes  = 1,
             drop   = 1,
             lr     = 1,
             epochs = 1,
             batchs = 1):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(2**nodes, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batchs, epochs=epochs, verbose=1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    return loss, accuracy
# 2. SETUP HYPERPARAMETER SEARCH SPACE
parameters = [
    sherpa.Continuous('lr',   [1e-4, 1e-2], 'log'),
    sherpa.Continuous('drop', [0.2, 0.5]),
    sherpa.Choice('epochs',   [5, 6, 7, 8, 9, 10]),
    sherpa.Choice('batchs',   [32, 64, 128, 256, 512, 1024]),
    sherpa.Discrete('nodes',  [5, 10])]
# 3. SETUP POPULATION SEARCH PARAMETERS
algorithm = sherpa.algorithms.PopulationBasedTraining(
    population_size=20,                                       # Number of random attempts in a generation
    num_generations=5)                                        # Best performing population of a generation move to next generation
    #perturbation_factors=(0.8, 1.2))                         # Local search arround a Continuous or Discrete hyperparameter by a quantity between these two values
    #parameter_range={'lr': [1e-6, 1e-1], 'drop': [0.2, 0.5]})# lower and upper limit what should not be crossed from while performing perturbation
# 4. METRIC IS BETTER LOW OR HIGH?
study = sherpa.Study(
    parameters=parameters,
    algorithm=algorithm,
    lower_is_better=False)
for trial in study:
    generation = trial.parameters['generation']
    load_from  = trial.parameters['load_from']
    print('-'*55)
    print('Generation {} - Population {}'\
    .format(generation, trial.parameters['save_to']))
    # 5. ADD HYPERPARAMETERS HERE
    loss, accuracy = TheModel(nodes=trial.parameters['nodes'],
                              drop=trial.parameters['drop'],
                              lr=trial.parameters['lr'],
                              epochs=trial.parameters['epochs'],
                              batchs=trial.parameters['batchs'])
    print('Test loss:  {} - Test accuracy:  {}'\
    .format(round(loss, 4), round(accuracy, 4)))
    study.add_observation(trial=trial,
                          iteration=generation,
                          objective=accuracy,
                          context={'loss': loss})
    study.finalize(trial=trial)
print('\n=====BEST RESULTS=====')
results = study.get_best_result()
for key in results: print('{:>10}: {}'.format(key, results[key]))
