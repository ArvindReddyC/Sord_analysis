def encapsule(df):
    display(df.head(5))
    print( f'shape of df = {df.shape}' )
    return df

def tran(x):
    y = x[1:-1]
    return y.split(',')

def only_activities( df , list_of_activities_to_include ): 
    return df[ df.Activity.isin( list_of_activities_to_include ) ]


def data_wrangle( df ):
    sample = df[['timestamp','raw_sensor_data']];
    sample['parameters'] = sample.raw_sensor_data.apply(tran);
    main = pd.DataFrame( sample.parameters.to_list() , columns = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11'] );
    main['timestamp'] = sample['timestamp'];
    main.drop( 'v1' , axis = 1 , inplace = True );
    main.set_index('timestamp' , inplace=True);
    return main;


def removeprocess(x):
    try:
      main , *_  = x.split(' ')
    except (ValueError,AttributeError):
      return x
    return main

def val_counts(df):
    return df['Activity'].value_counts()
'''
Will t
'''
def groupby_avarage(activity_df):
    activity_df['Time'] =  activity_df.Time.astype(int)
    means_df = activity_df.groupby('Time').mean()
    mode_of_activities = []
    for each in activity_df.groupby('Time'):
        mode_of_activities.append(each[1]['Activity'].mode()[0])
    means_df['Activities'] = mode_of_activities
    return means_df


def process_sheet(df, number):
    df = df.fillna(method='ffill')
    df.drop( ['SORD', 'AP time' , 'AP classification'] , axis = 1 , inplace = True )
    df = only_activities( df , ['Standing', 'Sitting', 'Reclining' , 'Walking', 'Sitting (outsrtetched legs)'] )
    df['User_Id'] = number 
    df['Activity'] = df.Activity.apply(removeprocess)
    df.Activity = df.Activity.replace( 'Reclining' , 'Sitting' )  
    display(val_counts(df))
    return df


def process_sheet_edit(df, number):
    df = df.fillna(method='ffill')
    display(val_counts(df))
    df['Activity'] = df.Activity.apply(removeprocess)
    # df.drop( ['SORD', 'AP time' , 'AP classification'] , axis = 1 , inplace = True )
    df = only_activities( df , ['Standing', 'Sitting', 'Reclining' , 'Walking', 'Sitting'] )
    df['User_Id'] = number 
    df.Activity = df.Activity.replace( 'Reclining' , 'Sitting' )  
    display(val_counts(df))
    return df


def drop_un(df):
  try:
    df.drop( ['SORD', 'AP time' , 'AP classification'] , axis = 1 , inplace = True )
  except:
    return df 
  return df


'''
Written for Excel file clearning
'''
#Steps to convert a UTC timezone to AUD time zone. 
def change_time(some_df):
  some_df = some_df.reset_index()
  #display(some_df.head(5))
  some_df['timestamp'] = pd.to_datetime(some_df['timestamp'] , utc= True )
  #display(some_df.dtypes)
  some_df['timestamp'] = some_df['timestamp'].dt.tz_convert('Australia/Melbourne')
  some_df.set_index('timestamp', inplace = True)
  some_df[some_df.columns.drop('Activity')] = some_df[some_df.columns.drop('Activity')].applymap( pd.to_numeric )
  return some_df


'''
Written by Poojith
'''
import pandas as pd
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
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
    main = pd.DataFrame( sample.parameters.to_list() , columns = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11'] );
    main['timestamp'] = sample['timestamp'];
    main.drop( 'v1' , axis = 1 , inplace = True );
    main.set_index('timestamp' , inplace=True);
    return main;

'''
Use data_wrangle_json  when we have data coming in as JSON
'''
def data_wrangle_json( df ):
    main = pd.DataFrame( df.raw_sensor_data.to_list() , columns = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11'] );
    main['timestamp'] = df['timestamp'];
    main.drop( 'v1' , axis = 1 , inplace = True );
    main.set_index('timestamp' , inplace=True);
    return main;

'''
Creating windows of size 35 with step = 1 (Rolling window)
'''
def create_segments_and_labels(df, time_steps, step):

    # x, y, z acceleration as features
    N_FEATURES = 10
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['v3'].values[i: i + time_steps]
        ys = df['v4'].values[i: i + time_steps]
        zs = df['v5'].values[i: i + time_steps]
        angle = df['v2'].values[i: i + time_steps]
        u5 = df['v6'].values[i: i + time_steps]
        u6= df['v7'].values[i: i + time_steps]
        u7= df['v8'].values[i: i + time_steps]
        u8= df['v9'].values[i: i + time_steps]
        u9= df['v10'].values[i: i + time_steps]
        u10= df['v11'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        segments.append([xs, ys, zs,angle,u5,u6,u7,u8,u9,u10 ])

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    reshaped = reshaped_segments.reshape(-1,350)
    return reshaped

#pass the model when u call this mehod.

'''
how to call the model, model should be a global variable and should be assigned when we run the flask server
from tensorflow import keras
model = keras.models.load_model('/content/saved_model/my_model_2')
'''

def main(data ):
  #sample = pd.DataFrame.from_dict(pd.DataFrame.from_dict(data[0]['data']))
  #main = data_wrangle_json(sample)
  # sample = pd.read_csv('sord.csv')
  # main = data_wrangle(sample);
  reshaped = create_segments_and_labels(data,35,1)
  model = keras.models.load_model('/content/saved_model/my_model_2')
  currently_activity = ''
  number_of_windoews = len(data) - 35
  from_ = data.loc[0,'timestamp']
  Clsses = ['sit', 'std', 'wlk']
  final_list = []
  for each in tqdm(range(number_of_windoews)):
    y_hat =  model.predict(reshaped[each].reshape(-1,350))
    y_hat  = np.argmax(y_hat)
    y_hat = Clsses[y_hat]
    if((y_hat != currently_activity) & (y_hat != '') ):
      final_list.append( { 'start_time': from_ , 'end_time':data.loc[each,'timestamp'] , 'activity': y_hat  } )
      currently_activity = y_hat
      from_ = data.loc[each,'timestamp']
  return final_list


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






