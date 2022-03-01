import pandas as pd
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from scipy import stats
from tqdm import tqdm


'''
Fucntion to split a string at ',' used in data_wrangle
'''
def tran(x):
    y = x[1:-1]
    return y.split(',')

'''
Use data_wrangle  when we have data coming in as CSV
'''
def data_wrangle( df ):
    sample = df[['timestamp','raw_sensor_data']];
    sample['parameters'] = sample.raw_sensor_data.apply(tran);
    main = pd.DataFrame( sample.parameters.to_list() , columns = ['Angle', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz' ] );
    main['timestamp'] = sample['timestamp'];
    main.drop( 'v1' , axis = 1 , inplace = True );
    main.set_index('timestamp' , inplace=True);
    return main;

'''
Use data_wrangle_json  when we have data coming in as JSON
'''
def data_wrangle_json( df ):
    main = pd.DataFrame( df.raw_sensor_data.to_list() , columns = ['time','Angle', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz' ] );
    main['timestamp'] = df['timestamp'];
    main.drop( 'time' , axis = 1 , inplace = True );
    main.set_index('timestamp' , inplace=True);
    return main;

'''
Creating windows of size 35 with step = 1 (Rolling window)
'''
def create_segments_and_labels(df, time_steps, step):
    N_FEATURES = 10
    segments = []
    labels = []
    indexes = df.index
    start_time = []
    for i in range(0, len(df) - time_steps, step):
        angle = df['Angle'].values[i: i + time_steps]
        ax = df['ax'].values[i: i + time_steps]
        ay = df['ay'].values[i: i + time_steps]
        az = df['az'].values[i: i + time_steps]
        gx = df['gx'].values[i: i + time_steps]
        gy=  df['gy'].values[i: i + time_steps]
        gz=  df['gz'].values[i: i + time_steps]
        mx =  df['mx'].values[i: i + time_steps]
        my = df['my'].values[i: i + time_steps]
        mz  = df['mz'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment 
        segments.append([angle, ax , ay , az , gx , gy , gz , mx , my , mz ])
        start_time.append( df.index[i] )
    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    return (reshaped_segments,start_time)

#pass the model when u call this mehod.

'''
how to call the model, model should be a global variable and should be assigned when we run the flask server
from tensorflow import keras
model = keras.models.load_model('/content/saved_model/my_model_2')
'''

def main(data , model ):
  sample = pd.DataFrame.from_dict(data['data'])
  main = data_wrangle_json(sample)
  # sample = pd.read_csv('sord.csv')
  # main = data_wrangle(sample);
  main.drop(main.iloc[np.where( main == '-' )[0] , ].index, inplace=True)
  main.Angle = (180 -  main.Angle)
  
  reshaped,start_time = create_segments_and_labels(main,64,32)
  if(reshaped.shape[0] < 1):
      print('Input  is not sufficient to make a prediction ')
      return 0
  
  #model = keras.models.load_model('/content/saved_model/my_model_2')
  
  predictions =  model.predict( reshaped )
  preds_df = pd.DataFrame(predictions)
  preds_df.fillna( method = 'ffill' , inplace = True )
  print(f'shape of predictions = {preds_df.shape}')
  preds_df.dropna( how = 'all' , inplace = True  )
  predds = []
  for i in  preds_df.index:
    predds.append(np.argmax( preds_df.loc[i,] ))
  
  arr = np.asarray( predds )
  indices = np.where( arr[:-1] != arr[1:])[0]
  indices = np.insert(indices, 0, -1, axis=0)
  indices = np.insert(indices, len(indices), len(predds) - 2 , axis=0)
  listt = []
  for i in range(0, len(indices) - 1  , 1 ):
    listt.append( { 'start_time' : start_time[ indices[i] + 1 ],  'end_time' : start_time[ indices[(i+1)] + 1 ] , 'activity' :  predds[ indices[i] + 1 ] } )
  temp =  pd.DataFrame( listt )
  temp.activity = temp.activity.map( { 0 : 'Sitting' , 1 :'Standing' , 2 : 'Walking' } )
  final_json =  temp.to_dict( orient = 'records' )
  return final_json


  # currently_activity = ''
  # number_of_windoews = len(sample) - 35
  # from_ = sample.loc[0,'timestamp']
  # Clsses = ['sit', 'std', 'wlk']
  # final_list = []
  # for each in tqdm(range(number_of_windoews)):
  #   y_hat =  model.predict(reshaped[each].reshape(-1,350))
  #   y_hat  = np.argmax(y_hat)
  #   y_hat = Clsses[y_hat]
  #   if((y_hat != currently_activity) & (y_hat != '') ):
  #     final_list.append( { 'start_time': from_ , 'end_time':sample.loc[each,'timestamp'] , 'activity': y_hat  } )
  #     currently_activity = y_hat
  #     from_ = sample.loc[each,'timestamp']
  # return final_list


#How to call the function.?
# main([{
#   "data": [
#     {
#       "raw_sensor_data": list(np.random.randint(10 , size = 11)),
#       "timestamp": "2022-02-10"    }
#   ],
#   "mode": "MANUAL",
#   "location": {
#     "latitude": 0,
#     "longitude": 0  },
#   "deviceStatus": "CONNECTED",
#   "actualActivity": "SITTING"} , {
#   "data": [
#     {
#       "raw_sensor_data": list(np.random.randint(10 , size = 11)),
#       "timestamp": "2022-02-10"    }
#   ],
#   "mode": "MANUAL",
#   "location": {
#     "latitude": 0,
#     "longitude": 0  },
#   "deviceStatus": "CONNECTED",
#   "actualActivity": "SITTING"}] )


main([{
  "data": [
    {
      "raw_sensor_data": list(np.random.randint(10 , size = 11)),
      "timestamp": "2022-02-10 01:01:00"    },
      {
      "raw_sensor_data": list(np.random.randint(10 , size = 11)),
      "timestamp": "2022-02-10 01:02:00"    },
      {
      "raw_sensor_data": list(np.random.randint(10 , size = 11)),
      "timestamp": "2022-02-10 01:03:00"    },
      {
      "raw_sensor_data": list(np.random.randint(10 , size = 11)),
      "timestamp": "2022-02-10 01:04:00"    }
  ],
  "mode": "DEVICE",
  "location": {
    "latitude": 0,
    "longitude": 0  },
  "deviceStatus": "CONNECTED",
  "actualActivity": "SITTING"}] , 'model' )

'''
return Json in the below Format

[{'activity': 'Sitting',
  'end_time': '2022-02-25 01:59:40.601',
  'start_time': '2022-02-25 01:44:00.001'},
 {'activity': 'Walking',
  'end_time': '2022-02-25 01:59:41.774',
  'start_time': '2022-02-25 01:59:40.601'},
 {'activity': 'Sitting',
  'end_time': '2022-02-25 02:07:37.137',
  'start_time': '2022-02-25 01:59:41.774'},
 {'activity': 'Standing',
  'end_time': '2022-02-25 02:21:14.144',
  'start_time': '2022-02-25 02:07:37.137'},
 {'activity': 'Sitting',
  'end_time': '2022-02-25 02:32:15.061',
  'start_time': '2022-02-25 02:21:14.144'},
 {'activity': 'Walking',
  'end_time': '2022-02-25 02:40:13.197',
  'start_time': '2022-02-25 02:32:15.061'}]
'''