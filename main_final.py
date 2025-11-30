import pandas as pd
import numpy as np
import os
from numpy import sin, cos, sqrt
from numpy import arctan2 as atan2
from numpy import arccos as acos
import warnings
from scipy.signal import butter, lfilter
from model_final import run_models

np.seterr(divide='ignore', invalid='ignore')
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

bad_cols=['start','end','idx','flight_id','takeoff','landed']

def butter_lowpass_filter(data, cutoff, fs, order=5):
    #A butterworth filter to smooth the data a bit
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def filters_and_interpolation(dfc):
    #Interpolates and then filters.
    #Ideally you model the flightpath based on other flights
    
    df=dfc.copy()
    
    #un"loop" heading so you can do meaningful calculations, not go 360 to zero
    df.loc[~df.track.isna(),'track']=np.unwrap(df.track[~df.track.isna()],period=360)
    
    #%%resampling to around every second
    df['expand_timestamps']=(-df.timestamp.diff(-1).dt.total_seconds()//1).fillna(0)
    timestamps=df.apply(lambda x:[x.timestamp+pd.Timedelta(seconds=xx) for xx in np.arange(1,1+x.expand_timestamps)],axis=1)
    timestamps=timestamps.explode()
    timestamps=timestamps[~timestamps.isna()]
    timestamps.name='timestamp'
    df=pd.concat([df,timestamps])
    df.flight_id=df.flight_id.iloc[0]
    df.index=df.timestamp
    df=df.drop(['expand_timestamps','timestamp'],axis=1)
    df=df.sort_values('timestamp')
    df=df.reset_index()
    
    #%%interpolate   
    todo=['track','vertical_rate','latitude','longitude','altitude','groundspeed']
    for col in todo:
        #lazy method. For climb and descent this is not correct. 
        df[col]=df[col].interpolate()
  
    #%%filter
    #filter specs chosen by experiment
    cutoff=1
    fs=60
    order=5
    
    for col in todo:
        if(df[col].dtype==float):
            #ignore NA
            no_nan=df[col][~df[col].isna()].copy()
            #filter from left side, and reversed direction. Filter starts at zero, add starting value later
            y = butter_lowpass_filter(no_nan-no_nan.iloc[0], cutoff, fs, order)+no_nan.iloc[0]
            yr = butter_lowpass_filter((no_nan-no_nan.iloc[-1])[::-1], cutoff, fs, order)[::-1]+no_nan.iloc[-1]
            yt=(y+yr)/2
            df.loc[~df[col].isna(),col]=yt
    return df

def distance_between_points(p1, p2, unit='meters', func='haversine'):
    #re-used from last year's challenge :)
    #sped up the code/method in the link by using numpy for vectorization instead of default math library
    """ This function computes the distance between two points in the unit given in the unit parameter.  It will
    calculate the distance using the haversine unless the user specifies haversine to be False.  Then law of cosines
    will be used
    :param p1: tuple point of (lon, lat)
    :param p2: tuple point of (lon, lat)
    :param unit: unit of measurement. List can be found in constants.eligible_units
    :param haversine: True (default) uses haversine distance, False uses law of cosines
    :return: Distance between p1 and p2 in the units specified.
    https://github.com/seangrogan/great_circle_calculator
    """
    
    p11 = np.radians(p1)
    p22 = np.radians(p2)
    lon1, lat1 = p11.longitude, p11.latitude
    lon2, lat2 = p22.longitude, p22.latitude
    r_earth = 6371000

    if (func == 'haversine'):
        # Haversine
        d_lat, d_lon = lat2 - lat1, lon2 - lon1
        a = sin(d_lat / 2) * sin(d_lat / 2) + cos(lat1) * cos(lat2) * sin(d_lon / 2) * sin(d_lon / 2)
        a = np.minimum(1, a)
        c = 2 * atan2(sqrt(a), sqrt((1 - a)))
        dist = r_earth * c
        return dist

    # Spherical Law Of Cosines
    dist = acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1)) * r_earth
    return dist


def flight_train_fixes(data):
    #Some more data wrangling
    #add airport data
    df_train=data['flight_list'].merge(data['fuel'],how='left',on='flight_id')
    df_train=df_train.merge(data['apt'],left_on='origin_icao',right_on='icao',how='left')
    df_train=df_train.merge(data['apt'],left_on='destination_icao',right_on='icao',how='left')
    cols_keep=['flight_date', 'aircraft_type', 'takeoff', 'landed', 'origin_icao',
           'destination_icao', 'flight_id',
           'idx', 'start', 'end', 'fuel_kg', 'longitude_x', 'latitude_x',
           'elevation_x', 'longitude_y', 'latitude_y', 'elevation_y']
    df_train=df_train[cols_keep]

    #fix timezones
    for col in ['landed','takeoff','start','end']:
        if(not df_train[col].dt.tz):
            df_train[col]=df_train[col].dt.tz_localize('UTC')
    
    return df_train

def flight_data_parser(df,dft):
    df=df.sort_values('timestamp')
    df.timestamp=df.timestamp.dt.tz_localize('UTC')
    flight_id=df.flight_id.iloc[0]
    df_train=dft[dft.flight_id==flight_id].copy()

    #add airport data to start and end of the flightdata, sometimes this is completely missing
    df_startend=pd.concat([df.iloc[:1],df.iloc[-1:]])
    df_startend.index=[-1,len(df)]
    df_startend['altitude']=[df_train.elevation_x.iloc[0],df_train.elevation_y.iloc[0]]
    df_startend['timestamp']=[df_train.takeoff.iloc[0],df_train.landed.iloc[0]]
    df_startend=df_startend[['longitude','latitude','altitude','timestamp','flight_id']]
    df=pd.concat([df,df_startend])
    df.index+=1
    df=df.sort_index()
    
    #possible duplicates in data
    df=df.drop_duplicates('timestamp',keep='first')
    
    #do some filter stuff
    df=filters_and_interpolation(df)
    
    df['tdiff']=abs(df.timestamp.diff(1)).fillna(abs(df.timestamp.diff(-1)))
    df.tdiff=df.tdiff.dt.total_seconds()  #should be ~1

    #distance
    df['dist']=distance_between_points(df[['longitude','latitude']],df[['longitude','latitude']].shift(1)).fillna(distance_between_points(df[['longitude','latitude']],df[['longitude','latitude']].shift(-1)))
    
    #%%main loop for data creation from the flight trajectories to model variables
    dist_cumsum=[]
    dist=[]
    alt1=[]
    alt2=[]
    track=[]
    vertical_rate=[]
    TAS=[]
    CAS=[]
    source=[]
    groundspeed=[]
    altitude=[]
    dheading=[]
    cols_todo=['groundspeed','track','vertical_rate','TAS','CAS','altitude']

    for i,row in df_train.iterrows():
        #take section of the flight trajectory
        _df=df[(df.timestamp>=row.start) & (df.timestamp<row.end)]
    
        if(_df.empty):
            #time error, slice is before or after trajectory data
            print('error | '+row.flight_id+' | '+str(row.idx))
            
            before=df[(df.timestamp<row.start)].iloc[-1:]
            after=df[(df.timestamp>row.end)].iloc[:1]
            
            #it might be a flight with only 1 data sample somehow..
            if(before.empty):
                before=after
            if(after.empty):
                after=before

            alt1+=[before.altitude.iloc[0]]
            alt2+=[after.altitude.iloc[0]]
            d=distance_between_points(before[['longitude','latitude']],after[['longitude','latitude']])
            d=d.sum()
            dist+=[d]
            dist_cumsum+=[d]
            
            dheading+=[(before.track-after.track).abs().sum()]
            
            for col in cols_todo+['source']:
                locals()[col]+=[np.nan]
        else:
            #altitude at start
            alt1+=[_df.altitude.iloc[0]]
            
            #altitude at end
            alt2+=[_df.altitude.iloc[-1]]
            
            #distance flown in this slice
            d=_df.dist.sum()
            dist+=[d]
            
            #change in heading
            dheading+=[(_df.track.diff()).abs().sum()]
            
            #distance flown from takeoff up to and including this slice 
            _df2=df[(df.timestamp<row.end)].copy()
            dist_cumsum+=[_df2.dist.sum()]
            
            #other variables, just take mean
            for col in cols_todo:
                locals()[col]+=[(_df[col][~_df[col].isna()]).mean()]
            
            #most common data source
            _s=_df.source.value_counts()
            if(_s.empty):
                source+=[np.nan]
            else:
                source+=[_s.iloc[0]]
                
    #%%
    #section length
    df_train['tdelta']=((df_train.end-df_train.start).dt.total_seconds())
    
    df_train['dist']=dist
    df_train['alt1']=alt1
    df_train['alt2']=alt2
    df_train['dheading']=dheading
    df_train['dist_cumsum']=dist_cumsum
    
    #total flown distance of that flight
    df_train['total_dist']=df.dist.sum()
    
    for col in cols_todo:
        df_train[col]=locals()[col]
        
    #groundspeed and vertical speed
    df_train['velocity']=(np.sqrt(df_train.vertical_rate**2+(df_train.groundspeed*101.2686)**2))        
    
    #Should be vertical rate during climb. Climb is not nicely defined.
    #Basically anything at the beginning ofthe flight, with low altitude and positive vertical rate.
    df_train['climbing_velocity']=df.vertical_rate[(df.altitude<10000) & (df.index<=(len(df)//2)) & (df.vertical_rate>5)].mean()
    
    return df_train,df


#create data from flight trajectories, save output to use for modelling
#Takes an hour or two on my pc, single thread
for _dir in ['train_data','rank_data','rank2_data']:   
    data={}
    for file in os.listdir(f'data/{_dir}/'):
        if(file.endswith('.parquet')):
            data[file.split('.')[0]]=pd.read_parquet(f'data/{_dir}/'+file)
    
    flights=[]
    
    files=os.listdir(f'data/{_dir}/flights/')
    
    df_train=flight_train_fixes(data)
    for i,file in enumerate(files):
        if(i % 10==0):
            print(f'{i}/{len(files)}')
            
        df=pd.read_parquet(f'data/{_dir}/flights/'+file)

        df_train2,df=flight_data_parser(df,df_train.copy())
        flights+=[df_train2]
        
    df_flights=pd.concat(flights).reset_index(drop=True)
    df_flights.to_csv(f'data/{_dir}/flightdata.csv',index=False)

#make model, predict and save as parquet
num='final'
df_train=run_models(num)
