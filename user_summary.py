import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters

class UserSummary:
    def __init__(self, user_id):
        register_matplotlib_converters()
        # load required data
        sm = pd.read_csv("sensor_match.csv",names=['user_id','time','id','glucose_reading','sensor'])
        self.user=self.clean_data(sm[sm.user_id==user_id])
        self.user_id=user_id
        self.trends={}
    '''
        Return summary statistics about sensor sensor glucose data
        - In particular, count (# timestamps), mean, standard deviation, quartiles
    '''
    def summarize(self):
        #group by sensor used and get summary statistics
        stats=self.user.groupby('sensor').describe()
        print('Summary Statistics for User ' + str(self.user_id))
        print(stats)
        print()
    
    '''
        Return regression statistics about sensor glucose data
         - In particular, correlation (r^2), y-intercept, x-intercept
        Plot the regression lines with data as well
    '''
    def regression(self):
        # partition timestamps/data into sensor used
        sensor_groupby = self.user.groupby('sensor')
        for group in sensor_groupby.groups:
            # remove sensor column (unnecessary)
            tsdata=sensor_groupby.get_group(group).drop(columns=['sensor'])
            # plot the regression plot
            sns.regplot(x=np.arange(0, len(tsdata)),y=tsdata.glucose_reading)
            # set x-axis (time), y-axis (glucose value)
            x=np.array(np.arange(0, len(tsdata))).reshape((-1, 1))
            y=np.array(tsdata.glucose_reading)
            # conduct linear regression
            model = LinearRegression().fit(x, y)
            # update trends accordingly
            self.trends[group]={'r^2': model.score(x, y), 'y_0':model.intercept_, 'm':model.coef_, 'model':model}
            # print regression statistics (r^2 correlation score, y-intercept, slope)
            print(str(group)+': r^2: '+str(model.score(x, y))+' y_0: '+str(model.intercept_)+' m: '+str(model.coef_))
        # graph the regression plots
        plt.show()
    
    def detrend(self):
        return None

    def clean_data(self, user):
        # find invalid timestamps in user data
        invalid_times=[]
        for t in user.time:
            try: 
                pd.to_datetime(t)
            except ValueError:
                invalid_times.append(t)

        # filter out bad time stamps
        user=user[~user.time.isin(invalid_times)]
        # create time-series data
        user.index=pd.to_datetime(user.time)
        # drop time, user_id columns since it's now index, useless
        user=user.drop(columns=['time','user_id','id'])
        # convert columns to proper dtype
        user=user.astype(int)
        #print(user.dtypes)
        return user

    def plot(self):
        #plot time series data
        t=self.user.index #timestamps
        bg=self.user.glucose_reading #sensor glucose measurements
        s=self.user.sensor #sensor used
        fig, axs = plt.subplots(2)
        axs[0].scatter(t,bg)
        axs[1].scatter(t,s)
        plt.show()