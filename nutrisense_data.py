import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# load required data
sm = pd.read_csv("sensor_match.csv",names=['user_id','time','id','value','sensor'])

# isolate one user (i.e. user_id=7)
user=sm[sm.user_id==16]

# find invalid timestamps in user data
invalid_times=[]
for t in user.time:
    try: 
        pd.to_datetime(t)
    except ValueError:
        invalid_times.append(t)

# filter out bad data
user=user[~user.time.isin(invalid_times)]
# create time-series data
user.index=pd.to_datetime(user.time)
# drop time, user_id columns since it's now index, useless
user=user.drop(columns=['time','user_id','id'])
# convert columns to proper dtype
user=user.astype(int)
#print(user.dtypes)

#plot time series data
#correspondence with sensors

t=user.index #timestamps
bg=user.value #sensor glucose measurements
s=user.sensor #sensor used
'''
fig, axs = plt.subplots(2)
axs[0].scatter(t,bg)
axs[1].scatter(t,s)
plt.show()
'''

#group by sensor used
sensor_groupby=user.groupby('sensor')
sensor_group_stats=sensor_groupby.describe()
trends={}
for group in sensor_groupby.groups:
    tsdata=sensor_groupby.get_group(group).drop(columns=['sensor'])
    sns.regplot(x=np.arange(0, len(tsdata)),y=tsdata.value)
    x=np.array(np.arange(0, len(tsdata))).reshape((-1, 1))
    y=np.array(tsdata.value)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    y0=model.intercept_
    m=model.coef_
    print(str(group)+': r^2: '+str(r_sq)+' y_0: '+str(y0)+' m: '+str(m[0]))
    trends[group]=model
