from collections import defaultdict
import math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

TRAIN_DATA_PATH = './train.csv'
TEST_DATA_PATH = './test.csv'

# '-1' means read all the gauge readings
NUM_TRAINING_GAUGE_READINGS = -1
NUM_TESTING_GAUGE_READINGS = -1
MAX_NUM_RADAR_READINGS_PER_HOUR = 30
# the dimension of the data (22 columns from the file, plus one make-up column)
X_DIM = 23

def load_rain_data(path):
  with open(path, 'r') as f:
    # if there is 24 columns, the file contains the training data (with the 
    # gauge reading), otherwise it contains the testing data (23 columns)
    is_training_file = len(f.readline().split(','))  == 24  
    print(f'DEBUG is_training_file: {is_training_file}')

    # key: gauge_reading_id
    radar_readings_of = defaultdict(list)
    gauge_readings = dict()
    last_gauge_reading_id = 0
    # group the lines by gauge_reading_id (the first column)
    for line in f:
      line = line.strip()
      # skip empty line
      if not line:
        continue
      cols = line.split(',')
      # some lines just have commas, also skip those lines
      if len(cols) == 0:
        continue

      gauge_reading_id = int(cols[0])
      # stop if we have read NUM_TRAINING_GAUGE_READINGS gauge readings when training
      if (is_training_file
            and NUM_TRAINING_GAUGE_READINGS != -1
            and len(gauge_readings.keys()) == NUM_TRAINING_GAUGE_READINGS
            and gauge_reading_id != last_gauge_reading_id):
        break
      # stop if we have read NUM_TESTING_GAUGE_READINGS gauge readings when testing
      if (not is_training_file
            and NUM_TESTING_GAUGE_READINGS != -1
            and len(radar_readings_of.keys()) == NUM_TESTING_GAUGE_READINGS
            and gauge_reading_id != last_gauge_reading_id):
        break
      last_gauge_reading_id = gauge_reading_id

      # read the gauge reading if the file is a training file
      if is_training_file:
        gauge_reading = float(cols[-1])
        assert(gauge_reading_id not in gauge_readings
                or gauge_readings[gauge_reading_id] == gauge_reading)
        gauge_readings[gauge_reading_id] = gauge_reading
        # drop the column
        cols = cols[:-1]

      # drop the gauge_reading_id column
      cols = cols[1:]

      # cast the items in cols to float, fill the missing data with zeros
      cols = [float(x) if x else 0. for x in cols]

      # scale some of the fields in cols (fields in a radar reading)
      # the first column in cols is 'minute', so no need to scale
      for i in range(2, 10):
        val = math.pow(math.pow(10, cols[i]/10)/200, 0.625)
      for i in range(14, 18):
        val = math.pow(10, cols[i]/10)
      
      # append the radar reading to radar_readings_of[gauge_reading_id]
      radar_readings_of[gauge_reading_id].append(cols)
      
  # turn radar_readings_of and gauge_readings in np.ndarray
  y = np.array([gauge_readings[gauge_reading_id]
                              for gauge_reading_id in sorted(gauge_readings.keys())],
               dtype=np.float32)
                              
  # NOTE: 22 of the '24' columns in X are from the file, the last  
  #       column is a make-up column
  # NOTE: for those hours which do not have MAX_NUM_RADAR_READINGS_PER_HOUR
  #       readings, readings with zero values are added so that all the
  #       gauge readings have MAX_NUM_RADAR_READINGS_PER_HOUR radar
  #       readings
  X = np.zeros((len(radar_readings_of.keys()), MAX_NUM_RADAR_READINGS_PER_HOUR, X_DIM), np.float32)
  for (i, gauge_reading_id) in enumerate(sorted(radar_readings_of.keys())):
    radar_readings = radar_readings_of[gauge_reading_id]
    # add a make-up column: no. of radar readings for this gauge_reading_id
    num_radar_readings = len(radar_readings)
    for radar_reading in radar_readings:
      radar_reading.append(num_radar_readings)
    X[i,:num_radar_readings,:] = np.array(radar_readings)

  print(f'DEBUG no. of gauge readings read: {len(gauge_readings.keys())}')
  return X, y
      
def main():
  # load the training data
  X, y = load_rain_data('./train.csv')
  print(f'DEBUG X: {X.shape}')
  print(f'DEBUG y: {y.shape}')
  
  # setup the model
  model = Sequential()
  model.add(GRU(35, input_shape=(MAX_NUM_RADAR_READINGS_PER_HOUR, X_DIM),
                activation='sigmoid'))
  model.add(Dense(1, activation='linear'))
  model.compile(loss='mean_absolute_error', optimizer='rmsprop')
  
  # train the model
  kf = KFold(n_splits=50)
  for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    check_pointer = ModelCheckpoint(filepath=f'gru_{i}.hdf5', verbose=1,
                                   save_best_only=True)
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit(X_train, y_train, epochs=100, batch_size=256,
              validation_data=[X_test, y_test],
              callbacks=[check_pointer, early_stopper])

  # evaluate the model
  X_test, _ = load_rain_data('./test.csv')
  print(f'DEBUG X_test: {X_test.shape}')  
  y_pred = model.predict(X_test, batch_size=256, verbose=1)

  # read in the y_true
  df = pd.read_csv('./sample_solution.csv', index_col=0)
  y_true = df['Expected'].values[:y_pred.shape[0]]
  mse = mean_squared_error(y_true, y_pred)
  print(f'mse: {mse}')
    
if __name__ == '__main__':
  main()
