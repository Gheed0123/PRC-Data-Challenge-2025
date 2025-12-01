import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import requests
import matplotlib.dates as mdates

#Just a couple of plots of the leaderboard

#%%init
dfs=[]
ylim = 1200 #my bad
ymax = 300
ymin = 190
get=False#True

#%%get jsons
def jsonget(cursor=''):
    urlb='https://datacomp.opensky-network.org/api/competitions/71c49292-6139-425f-803a-52ee8730ba58/leaderboard?limit=100&cursor='
    s=requests.get(urlb+cursor)
    js=json.loads(s.content)
    return js

if(get):
    js=jsonget()
    cursor=js['nextCursor']
    lst=js['items']
    i=1
    while cursor:
        print(i)
        i+=1
        js=jsonget(cursor)
        cursor=js['nextCursor']
        lst+=js['items']
        
    df=pd.DataFrame(lst)
    df=df.drop_duplicates()
    df.to_csv("leaderboard.csv",index=False)
else:
    df=pd.read_csv('leaderboard.csv')
    
#%%fix data
df=df.sort_values(['score'])
df.processedAt=pd.to_datetime(df.processedAt)
df['version']=df.filename.str.extract('_v(.*)\.parquet').astype(int)
final_time=df.processedAt.max()

#%%plot data per team

for teamname,group in df.groupby('teamName',sort=False):
    fig,ax=plt.subplots()
    group=group.sort_values('processedAt')
    
    #ignore all non-improved scores
    group.score=group.score.cummin()
    
    #trace lines forward to final submit time
    #group=pd.concat([group,pd.DataFrame({'processedAt':[final_time],'score':[group.score.min()]})])
    
    #label creation
    # version=str(int(group.version.max()))
    # version=str(len(group))
    #best version number
    version=group.version[group.score==group.score.min()].iloc[0]
    
    label=teamname#+'_v'+str(version)+'_'+str(group.score.min().round(2))
    #label=teamname+'_'+str(group.score.min().round(2))
    
    #plot
    ax.semilogy(group.processedAt,group.score.values,label=label,marker='o',ms=3)
    
    #%%plot stuff
    ax.set_title(f'{label}')
    ax.grid()
    ax.grid(which="minor", color="0.9")
    ax.set_ylabel('score')
    ax.set_xlabel('submit date')
    ax.set_xlim(pd.Timestamp(year=2025,month=10,day=14),pd.Timestamp(year=2025,month=12,day=1))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for l in ax.get_xticklabels(which='major'):
        l.set(rotation=30, horizontalalignment='right')

    #ax.legend()
    plt.tight_layout()
    plt.savefig('leaderboard_plots/'+label)
#%%plot for all teams

fig,ax=plt.subplots()

for teamname,group in df.groupby('teamName',sort=False):
    group=group.sort_values('processedAt')
    
    #ignore all non-improved scores
    group.score=group.score.cummin()
    
    #trace lines forward to final submit time
    #group=pd.concat([group,pd.DataFrame({'processedAt':[final_time],'score':[group.score.min()]})])
    
    #label creation
    version=str(int(group.version.max()))
    version=str(len(group))
    label=teamname+'_v'+version+'_'+str(group.score.min().round(2))
    #label=teamname+'_'+str(group.score.min().round(2))
    
    #plot
    ax.semilogy(group.processedAt,group.score.values,label=label,marker='o',ms=3)
    ax.grid()
    ax.grid(which="minor", color="0.9")

#%%plot stuff
ax.set_title('team score over time')
ax.grid()
ax.set_ylabel('score')
ax.set_xlabel('submit date')
ax.set_ylim(ymin,ymax)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

#ax.legend(bbox_to_anchor=(1.1, 1.05),reverse=True)

#%%plot improvement steps per team
fig,ax=plt.subplots()

for teamname,group in df.groupby('teamName',sort=False):
    group=group.sort_values('processedAt')
    
    #ignore all non-improved scores, get the improvement
    group.score=group.score.cummin().diff().abs().fillna(0)
    
    #trace lines forward to final submit time
    #group=pd.concat([group,pd.DataFrame({'processedAt':[final_time],'score':[group.score.min()]})])
    
    #label creation
    label=teamname
    
    group=group[group.score>0]
    
    #plot
    ax.plot(group.processedAt,group.score.values,label=label,marker='o',ms=3)

#%%plot stuff
ax.set_title('improvement per team over time')
ax.grid()
ax.set_ylabel('score')
ax.set_xlabel('submit date')
ax.set_ylim(0,60)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')

#ax.legend(bbox_to_anchor=(1.1, 1.05),reverse=True)


#%%plot more
import numpy as np
fig,ax=plt.subplots()
maxlen=df.groupby('teamName').score.count().max()
for teamname,group in df.groupby('teamName',sort=False):
    group=group.sort_values('score',ascending=False)
        
    #label creation
    label=teamname
    
    #plot
    ax.semilogy(np.arange(0,maxlen+.01,((maxlen)/(len(group)-1))),group.score.values,label=label,marker='o',ms=3)

ax.set_title('submissions by score per team')
ax.grid()
ax.set_ylabel('score')
ax.set_ylim(ymin,ymax)

#ax.legend(bbox_to_anchor=(1.1, 1.05),reverse=True)

#%%top 10
import numpy as np
fig,ax=plt.subplots()
maxlen=df.groupby('teamName').score.count().max()
n=0
for teamname,group in df.groupby('teamName',sort=False):
    group=group.sort_values('score',ascending=False)
        
    #label creation
    label=teamname
    
    #plot
    ax.semilogy(np.arange(0,maxlen+.01,((maxlen)/(len(group)-1))),group.score.values,label=label,marker='o',ms=3)
    n+=1
    if(n==10):
        break

ax.set_title('top 10 teams, scores sorted')
ax.grid()
ax.set_ylabel('score')
ax.set_ylim(ymin,ymax)
ax.legend(bbox_to_anchor=(1.1, 1.05),reverse=True)

