import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def drop_this(df):  
    #remove illegal columns from model, these are only used for indexing purposes
    return df.drop(['fuel_kg','flight_id','idx'], axis=1).reset_index(drop=True).copy()

def make_model(df,cols):
    _df = drop_this(df.copy()[cols])
    
    #make model and fit
    model = XGBRegressor(n_estimators=250,learning_rate=.125)
    model=model.fit(_df, df.fuel_kg)

    #make prediction
    prediction = make_prediction(model,df,cols)
    return model,prediction

def make_prediction(model,df,cols):
    #make prediction from fitted model
    _df = drop_this(df.copy()[cols])
    prediction=model.predict(_df,output_margin=True)
    
    #refueling not included
    prediction[prediction<0]=0
    return prediction

def wrangle_data(df):
    #make variables and work the dataframe 
    
    #various FAA data per aircraft type
    #source: https://www.faa.gov/airports/engineering/aircraft_char_database
    aircraftdata=pd.read_excel(r"data\aircraft_data.xlsx")
    aircraftdata['aircraft_type']=aircraftdata.ICAO_Code
    df=df.merge(aircraftdata[['aircraft_type','FAA_Weight','MTOW_lb','Physical_Class_Engine','Num_Engines']],how='left')
    
    for col in df.columns:
        #convert dates to correct format
        if(df[col].dtypes=='object'):
            if(df[col].iloc[0].startswith('2025')):
                df[col]=pd.to_datetime(df[col],format='mixed')
              
    #apparent duration of flight
    df['flight_duration']=(df.landed-df.takeoff).dt.total_seconds()
    
    #track split to NS / EW two halves
    df['track_east_west']=np.sin(df.track/180*np.pi)
    df['track_north_south']=np.cos(df.track/180*np.pi)
        
    #time between start & end of the ranking section andthe takeoff or landing time
    #this can be negative for some reasons: asking about fuel use before the plane even takes off, 
    #or the start / landing times are not correct.
    df['time_until_landing_start']=(df.landed-df.start).dt.total_seconds()    
    df['time_since_takeoff_start']=(df.start-df.takeoff).dt.total_seconds()
    
    df['time_since_takeoff_end']=(df.end-df.takeoff).dt.total_seconds()
    df['time_until_landing_end']=(df.landed-df.end).dt.total_seconds()

    #from a timestamp to float 
    df['takeoff']=df.takeoff.dt.hour*3600+df.takeoff.dt.minute*60+df.takeoff.dt.second
    
    #average groundspeed from calculated distance
    df['groundspeed_from_dist_calc']=df.dist/df.tdelta

    #add fuel use base data
    #source: Appendix C  ICAO Carbon Emissions Calculator Methodology V13
    #https://icec.icao.int/Documents/Methodology%20ICAO%20Carbon%20Emissions%20Calculator_v13_Final.pdf
    #Note that this did not impact result by much, +-5 RMSE
    fueldata=pd.read_csv(r"data\fuel_use2.txt")
    
    #nm to m
    dist_m=fueldata.columns[1:].astype(int)*1852
    fueldata.columns=['aircraft_type']+list(dist_m)
    fueldata.index=fueldata.aircraft_type
    
    #Get the fuel use higher than the currently flown distance
    #Total flown distance was also tested, but like many things it is not in this final version
    col='dist_cumsum'
    
    fuels=[]
    dist_fuel=[]
    aircrafts=list(df.aircraft_type)
    for i,dist in enumerate(list(df[col])):
        d=dist_m[dist<dist_m]        
        if(any(d)):
            d=d[0]
        else:
            #Too far distance for the data
            d=dist_m[-1]
        dist_fuel+=[d]
        fuels+=[fueldata.loc[aircrafts[i],d]]
    
    df['fuel_burn']=fuels
    
    #adjust fuel use based on actual distance flown instead of value one higher
    df.fuel_burn=df.fuel_burn*df[col]/(dist_fuel)
    return df

def run_models(num):
    #%%training data
    print('loading')
    dfs=[]
    dirs=['train_data','rank_data','rank2_data']
    for _dir in dirs:
        dfs+=[pd.read_csv(f'data/{_dir}/flightdata.csv')]
    
    _dir=dirs[0]
    print(f'{_dir}')
    df=dfs[0].copy()
    
    df=wrangle_data(df.copy())
  
    #%%train set
    #these are the columns used in final model
    cols=['tdelta','vertical_rate','MTOW_lb','dist','takeoff','velocity','climbing_velocity',
          'time_until_landing_start','time_since_takeoff_start','time_since_takeoff_end',
          'alt1','alt2','elevation_y','track_east_west','track_north_south','dheading','flight_duration',
          'groundspeed_from_dist_calc','total_dist','fuel_burn',
          'fuel_kg','flight_id','idx']
    model,prediction=make_model(df.copy(),cols)
    
    #%%first test set
    _dir=dirs[1]
    print(f'{_dir}')
    df2=dfs[1].copy()
    df2=wrangle_data(df2)

    prediction2=make_prediction(model,df2.copy(),cols).round()
    
    df_save=pd.read_parquet(rf"data\{_dir}\fuel.parquet")
    df_save['fuel_kg']=prediction2
    df_save.to_parquet(f'data/{_dir}/submissions/genuine-zucchini_v{num}.parquet',index=False)
    
    #%%final test set
    _dir=dirs[2]
    print(f'{_dir}')
    df2=dfs[2].copy()
    df2=wrangle_data(df2)

    prediction3=make_prediction(model,df2.copy(),cols)

    df_save=pd.read_parquet(rf"data\{_dir}\fuel.parquet")
    df_save.fuel_kg=np.append(prediction2,prediction3)
    df_save=df_save.reset_index(drop=True)
    df_save.to_parquet(f'data/{_dir}/submissions/genuine-zucchini_v{num}.parquet',index=False)
    return dfs

if(__name__=='__main__'):
    num='final'
    df=run_models(num)   
    #final_solution score 219.5323
    #not the highest attained, but it is fine
