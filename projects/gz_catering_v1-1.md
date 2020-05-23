# Guangzhou Catering - Lunch Attendance Predictions

<font size="4">The attendance of a weekly catered lunch in Guangzhou, China varies week by week. When fewer people attend than anticipated, the business suffers waste from excess food, and when more people attend than the prepared food can accomodate, the extra business needs to be turned down and customers disappointed.<br>
    
***Can machine learning help predict lunch attendance to minimize these adverse effects of over- and under-preparing food?***</font>


[//]: # (This is for displaying in GitHub Pages)

![]<img src="images/gz_catering_image.jpg?raw=true"/>


## Background information
<font size="3">The weekend catering service serves lunch in the same venue that holds two weekly events in the morning. The attendance of these events also varies each week (sometimes by a lot), but  nearly all lunch attendees will go to one of the two morning events, so this data point will be correlated with the lunch attendance number we're trying to predict.<br>

Another consideration will be holidays and work days. In China, holidays could significantly influence a weekend event for two reasons. First, some festivals such as the Chinese New Year will see hundreds of millions of people traveling around the country to their hometowns. Secondly, smaller 3-day holidays can sometimes be held during the weekdays, shifting a work day or two to the weekend. So this data needs to be incorporated for an accurate prediction model.<br>

There will be other factors we'll consider later.</font>

## How can we evaluate our prediction model?
<font size="3">The method we'll use to determine the accuracy of our model is "MAE", or the "Mean Absolute Error". This means that we'll compare each prediction of our model with the actual value, giving us the "absolute error" for each prediction. We'll then average those together to see how far off, on average, the predictive model is. We'll also refer to the "Root Mean Squared Error" (RMSE) which will supplement our understanding of the model's performance*.</font>




___
>\* <font size="2">The RMSE squares each of the 'errors' (difference between each prediction and actual value), then adds those together and then takes the square root of that number. The purpose of this metric is to shed light on the distribution of the errors. For example, if Model A and Model B both have a Mean Absolute Error (average error) of 30 people, but Model A's RMSE is 40, where's Model B's RMSE is 35, this tells us that although both models average the same accuracy, Model B is more consistent in its results from one prediction to the next, whereas Model A's estimates vary more widely in accuracy.</font>

<font size="3">Thankfully, we have a benchmark against which to measure our model as the catering company received weekly human predictions for about half a year.</font>



[//]: # (This is for displaying in GitHub Pages)

![]<img src="images/human_predictions.PNG?raw=true"/>

<font size="4">We'll approach this problem in three steps:
>   1.  **Clean Data** - First, we'll clean up and augment the data 
>   2.  **Explore the Data** - Then we'll dive in and explore the data to guide us in building a model
>   3.  **Build Prediction Models** - Finally, we'll build some machine learning models to make predictions</font>


# #1 - Clean the Data

<font size="4">After importing the analysis packages and loading in the data, we see there are 519 records of data and 9 columns.</font>


```python
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates

lunch = pd.read_csv('data\\gz_catering\\gz_catering.csv')
lunch.shape
```




    (519, 9)



### Data types

<font size="4">The "Date" column was read in as an "object" type, so we'll want to convert this to datetime format for easier visualization later on. The rest of the columns are good to go.


```python
lunch.dtypes
```




    Date                         object
    EN_total                    float64
    EN_adults                   float64
    EN_children                 float64
    CN_total                    float64
    CN_adults                   float64
    CN_children                 float64
    Prediction_from_prior_wk    float64
    Lunch_total                 float64
    dtype: object




```python
lunch["Date"] = pd.to_datetime(lunch.Date)
```

### Missing values
<font size="4">Looking at the first few rows, we already see some missing data. "EN" refers to one of the two weekly events (an English-speaking event), and "CN" refers to the other Chinese-speaking event.<br>
    
We'll want to do something about these missing values, especially if the "Lunch_total" value is missing. Since this is the value we are trying to predict, any row without this data won't be helpful to our predictive model.


```python
lunch.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>EN_total</th>
      <th>EN_adults</th>
      <th>EN_children</th>
      <th>CN_total</th>
      <th>CN_adults</th>
      <th>CN_children</th>
      <th>Prediction_from_prior_wk</th>
      <th>Lunch_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-06-28</td>
      <td>184.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-07-05</td>
      <td>105.0</td>
      <td>99.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-07-12</td>
      <td>152.0</td>
      <td>145.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>94.0</td>
    </tr>
  </tbody>
</table>
</div>



<font size="4">Of these 519 rows of data, we see below that the CN event has a lot of missing data. The Lunch_total is also missing 104 rows. We'll drop these from the dataset.


```python
lunch.isnull().sum()
```




    Date                          0
    EN_total                     33
    EN_adults                    35
    EN_children                  41
    CN_total                    393
    CN_adults                   393
    CN_children                 394
    Prediction_from_prior_wk    490
    Lunch_total                 104
    dtype: int64



<font size="4">Now let's visualize the attendance of the two events. We can see:
>    - the second event "CN" is relatively new (from 2016 onward)
>    - the first event "EN" has seen decreased attendance since the start of the CN event
    
These two observations indicate there's been a meaningful shift in attendance patterns since the introduction of the CN event, which will be an important factor in whatever predictive model we choose later. We'll  use the 2016-onward data that includes both "EN" and "CN" events.


```python
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

fig, ax = plt.subplots()
ax.scatter(lunch.Date, lunch.EN_total, color="tab:gray", alpha=0.7, label="EN_total")
ax.scatter(lunch.Date, lunch.CN_total, color="tab:red", alpha=0.7, label="CN_total")
ax.legend(fontsize=12)
ax.set_title("Attendance of Two Events (CN & EN)", fontsize=14)
ax.set_ylabel("Attendance (people)", fontsize=11)
ax.tick_params(axis="both", which="both", length=0)
ax.set_ylim(0)
fig.set_size_inches(8, 6)
plt.show()
```


![png](output_14_0.png)


<font size="4">Now we'll drop from the dataset:
>- all rows that don't have a "Lunch_total", 
>- all rows that don't have either a "EN_total" or "CN_total", and 
>- all rows before the start of the "CN" event.


```python
lunch.dropna(subset=["Lunch_total"], how="any", inplace=True)
lunch.dropna(subset=["EN_total", "CN_total"], how="all", inplace=True)
lunch = lunch.loc[lunch.Date >= "2016/05/08", :]
lunch.shape
```




    (77, 9)



## Now, let's add some features that will help the model predict more accurately
### 1. Add Holidays & Workdays into the analysis
<font size="4">From our observations above, whether the weekend is a holiday or a make-up workday will likely be an important factor in weekend event attendance.


```python
workdays = pd.read_csv('data\\gz_catering\\workdays_holidays.csv')
workdays["Date"] = pd.to_datetime(workdays.Date)
lunch = pd.merge(lunch, workdays, how="inner", left_on="Date", right_on="Date")
```

### 2. Add some rolling attendance figures
<font size="4">We'll add in 2-week, 4-week and 8-week rolling attendance data for both Events and the Lunch attendance. 
><span style="color:maroon">**Note:** When adding new features like this, we need to be careful to not "leak" any data that the predictive model would not have access to in a real-world scenario. For example, if we were to use a rolling average which incorporated data from subsequent rows as part of the calculation, we would be leaking data that the model would not have access to in a real-world scenario. This **"data leakage"** would cause the model to artificially overperform on the training data, while predicting poorly on new data. 
    
In this case, however, we are using historical rolling attendance which we would have access to when making predictions.


```python
lunch["EN_prior_wk"] = lunch.EN_total.shift(periods=1)
lunch["CN_prior_wk"] = lunch.CN_total.shift(periods=1)
lunch["Lunch_prior_wk"] = lunch.Lunch_total.shift(periods=1)

COLS = ['EN_prior_wk', 'CN_prior_wk', 'Lunch_prior_wk']
new_names = ["EN", "CN", "Lunch"]
WKS = [2, 4, 8]
df = pd.DataFrame()

for wk in WKS:
    temp_df = lunch.loc[:, COLS].rolling(wk).mean()
    temp_df.columns = [x + '_roll_{}'.format(str(wk)) for x in new_names]
    df = pd.concat([df, temp_df], axis=1)
    
lunch = pd.concat([lunch, df], axis=1)
to_fill = [x for x in lunch.columns if x != "Prediction_from_prior_wk"]
lunch.loc[:, to_fill] = lunch.loc[:, to_fill].fillna(method="bfill")
```


```python
lunch.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>EN_total</th>
      <th>EN_adults</th>
      <th>EN_children</th>
      <th>CN_total</th>
      <th>CN_adults</th>
      <th>CN_children</th>
      <th>Prediction_from_prior_wk</th>
      <th>Lunch_total</th>
      <th>Is_Workday</th>
      <th>...</th>
      <th>Lunch_prior_wk</th>
      <th>EN_roll_2</th>
      <th>CN_roll_2</th>
      <th>Lunch_roll_2</th>
      <th>EN_roll_4</th>
      <th>CN_roll_4</th>
      <th>Lunch_roll_4</th>
      <th>EN_roll_8</th>
      <th>CN_roll_8</th>
      <th>Lunch_roll_8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-05-08</td>
      <td>241.0</td>
      <td>201.0</td>
      <td>40.0</td>
      <td>118.0</td>
      <td>105.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>234.0</td>
      <td>0</td>
      <td>...</td>
      <td>234.0</td>
      <td>238.0</td>
      <td>115.0</td>
      <td>217.0</td>
      <td>244.0</td>
      <td>109.0</td>
      <td>212.25</td>
      <td>234.5</td>
      <td>107.75</td>
      <td>215.375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-05-15</td>
      <td>235.0</td>
      <td>196.0</td>
      <td>39.0</td>
      <td>112.0</td>
      <td>101.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>200.0</td>
      <td>0</td>
      <td>...</td>
      <td>234.0</td>
      <td>238.0</td>
      <td>115.0</td>
      <td>217.0</td>
      <td>244.0</td>
      <td>109.0</td>
      <td>212.25</td>
      <td>234.5</td>
      <td>107.75</td>
      <td>215.375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-05-22</td>
      <td>251.0</td>
      <td>210.0</td>
      <td>41.0</td>
      <td>104.0</td>
      <td>94.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>219.0</td>
      <td>0</td>
      <td>...</td>
      <td>200.0</td>
      <td>238.0</td>
      <td>115.0</td>
      <td>217.0</td>
      <td>244.0</td>
      <td>109.0</td>
      <td>212.25</td>
      <td>234.5</td>
      <td>107.75</td>
      <td>215.375</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 26 columns</p>
</div>



<font size="4">We have all the features we need for now. Time to dig into, explore and get a feel for which features will likely help make the best prediction.

# #2 - Explore the Data


```python
import matplotlib as mpl
import seaborn as sns
```

<font size="4">First, let's plot out the relationship between Lunch attendance and the prior week's EN event, CN event and Lunch attendance. Since predictions will need to be made in advance of the weekend catering, the EN and CN attendances from that day will be unknown and thus can't be used in our predictive model.<br>
    
So let's see the relationship between the event attendances from one week to the Lunch attendance of the following week. 


```python
fig, ax = plt.subplots()
ax.plot(lunch.Date, lunch.EN_prior_wk, color="tab:gray", alpha=0.7, label="EN_prior_wk")
ax.plot(lunch.Date, lunch.CN_prior_wk, color="tab:gray", alpha=0.3, label="CN_prior_wk")
ax.plot(lunch.Date, lunch.Lunch_prior_wk, color="tab:red", alpha=0.2, label="Lunch_prior_wk")
ax.plot(lunch.Date, lunch.Lunch_total, color="tab:red", alpha=0.7, label="Lunch_total")
ax.legend(fontsize=10, loc=(0.45,0.02))
ax.set_title("Comparison of Prior Week Event Attendances \nand Lunch_total", fontsize=14)
ax.set_ylabel("Attendance (people)", fontsize=11)
ax.tick_params(axis="both", which="both", length=0)
fig.set_size_inches(8, 6)
plt.show()
```


![png](output_26_0.png)


<font size="4">Yikes. Let's make this somewhat more legible by 'straightening out' the red **<span style="color:maroon">Lunch_total line</span>** so we can just see the other three lines in relation to this line, which is the target we're trying to make predictions of.


```python
def graph1(lunch=lunch, c1="tab:gray", a1=0.7, a2=0.3):
    fig, ax = plt.subplots()
    ax.plot(lunch.Date, lunch.EN_prior_wk - lunch.Lunch_total, color="tab:gray", alpha=a1, label="EN_prior_wk")
    ax.plot(lunch.Date, lunch.CN_prior_wk - lunch.Lunch_total, color=c1, alpha=a2, label="CN_prior_wk")
    ax.plot(lunch.Date, lunch.Lunch_prior_wk - lunch.Lunch_total, color="tab:red", alpha=0.2, label="Lunch_prior_wk")
    ax.plot(lunch.Date, lunch.Lunch_total - lunch.Lunch_total, color="tab:red", alpha=0.7, label="Lunch_total")
    ax.legend(fontsize=10)
    ax.set_title("Comparison of Prior Week Event Attendances \nand Lunch_total", fontsize=14)
    ax.set_ylabel("Attendance (people)", fontsize=11)
    ax.tick_params(axis="both", which="both", length=0)
    fig.set_size_inches(8, 6)
    plt.show()
graph1()
```


![png](output_28_0.png)


<font size="4">That's better. We can see more clearly the other lines' relationships with this **Lunch_total** data that we're trying to predict.<br>
    
This seemingly spaghetti-stringed graphic reveals a telling trend. The **CN_prior_wk** data (in light grey) is trending upwards in relation to the **Lunch_total**, whereas the **EN_prior_wk** data is roughly staying at a proportionate relationship with the **Lunch_total**. This means a linear regression model will be more likely to pick up the stronger correlation of the EN event's attendance to the **Lunch_total**.  

<font size="4">Isolating the **CN_prior_wk** to a more contrasting color, we can see this trend clearer:


```python
graph1(lunch=lunch, c1="tab:green", a1=0.2, a2=0.7)
```


![png](output_31_0.png)


<font size="4">So we can see **EN_prior_wk** will be more strongly correlated with our target value **Lunch_total** than the CN event attendance.<br>
    
How about the make-up weekend workdays and holidays? Here are the make-up weekend workdays plotted on the raw **Lunch_total** attendance numbers:


```python
fig, ax = plt.subplots()
ax.plot(lunch.Date, lunch.Lunch_total, color="tab:red", alpha=0.7, label="Lunch_total")
ax.set_title("Lunch_total & Make-up Weekend Workdays", fontsize=14)
ax.set_ylabel("Attendance (people)", fontsize=11)
ax.tick_params(axis="both", which="both", length=0)
for day in list(lunch.loc[lunch.Is_Workday>0, "Date"]):
    if day == list(lunch.loc[lunch.Is_Workday>0, "Date"])[0]:
        ax.axvline(day, alpha=0.2, lw=5, label="Workday shifted to weekend")
    else:
        ax.axvline(day, alpha=0.2, lw=5)        
ax.legend(fontsize=12)
fig.set_size_inches(8, 6)
plt.show()
```


![png](output_33_0.png)


<font size="4">This chart illustrates a strong negative correlation between weekend **Is_Workday** and the **Lunch_total** attendance since every weekend that was shifted to a work weekend (due to national holidays) resulted in a huge dip in lunch attendance.<br>
    
Let's do the same for **Proximity_to_Major_Holiday**, such as the Spring Festival (Chinese New Year) and the National Holiday:


```python
fig, ax = plt.subplots()
ax.plot(lunch.Date, lunch.Lunch_total, color="tab:red", alpha=0.7, label="Lunch_total")
ax.set_title("Lunch_total & Major Holidays", fontsize=14)
ax.set_ylabel("Attendance (people)", fontsize=11)
ax.tick_params(axis="both", which="both", length=0)
holiday = list(lunch.loc[lunch.Proximity_to_Major_Holiday>0, "Date"])
for day in holiday:
    if day == holiday[0]:
        ax.axvline(day, alpha=0.2, color="tab:green", lw=5, label="Proximity to Major Holiday")
    else:
        ax.axvline(day, alpha=0.2, color="tab:green",  lw=5)        
ax.legend(fontsize=12)
fig.set_size_inches(8, 6)
plt.show()
```


![png](output_35_0.png)


<font size="4">Not surprisingly, **Proximity_to_Major_Holiday** also has a strong negative correlation with **Lunch_total**, especially during the Chinese New Year time.

## Correlations

<font size="4">Let's now examine the correlations between the below features which we'll be able to use in our predictive model and our target **Lunch_total** value:


```python
COLS = ['Lunch_total', 'Is_Workday', 'Is_Holiday', 'School_in_session', 'Proximity_to_Major_Holiday',
       'Proximity_to_Minor_Holiday', 'EN_prior_wk', 'CN_prior_wk',
       'Lunch_prior_wk', 'EN_roll_2', 'CN_roll_2', 'Lunch_roll_2', 'EN_roll_4',
       'CN_roll_4', 'Lunch_roll_4', 'EN_roll_8', 'CN_roll_8', 'Lunch_roll_8', 'Date']
lunch_select = lunch.loc[:, COLS]
corr = lunch_select.corrwith(lunch_select.Lunch_total).sort_values().drop("Lunch_total")
colors = corr.apply(lambda x: "tab:red" if x > 0 else "tab:blue")

fig, ax = plt.subplots()
ax.barh(corr.index.values, corr, color=colors, alpha=0.4)
ax.set_title("Correlation with Lunch_total", fontsize=14)
ax.tick_params(axis="both", which="both", length=0)
ax.yaxis.set_tick_params(pad=5)
for s in ["top", "bottom", "right", "left"]:
    ax.spines[s].set_color("white")
# ax.set_xlim(-0.6, 0.6)
fig.set_size_inches(8, 6)
plt.show()
```


![png](output_38_0.png)


<font size="4">As our analysis above showed, **Is_Workday** and **Proximity_to_Major_Holiday** are both strongly negatively correlated with the target value, whereas the EN event attendance is more strongly positively correlated. Also, the CN event attendance has the weakest correlation with **Lunch_total** of all these features, which makes sense as we saw from the line chart above.<br>
    
We'll refer to these correlations for our feature selections on the next step.

# #3 - Build Predictive Models


```python
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
```

## Split the data into "training" and "testing" splits

<font size="4">Since we want to have a direct comparison with the human predictions as a benchmark, we'll separate the data into *test* and *train* sets based on whether or not there's a human prediction.




```python
COLS = [
    'Is_Workday', 
    'Is_Holiday', 
    'School_in_session',
    'Proximity_to_Major_Holiday', 
    'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
    'CN_prior_wk',
    'Lunch_prior_wk', 
    'EN_roll_2',
    'CN_roll_2', 
    'Lunch_roll_2', 
    'EN_roll_4', 
    'CN_roll_4', 
    'Lunch_roll_4',
    'EN_roll_8', 
    'CN_roll_8', 
    'Lunch_roll_8'
]

def lunch_tts(df=lunch, COLS=COLS):

    X_train = df.loc[df.Prediction_from_prior_wk.isnull(), COLS]       
    X_test = df.loc[df.Prediction_from_prior_wk.notnull(), COLS]    
    y_train = df.loc[df.Prediction_from_prior_wk.isnull(), "Lunch_total"]
    y_test = df.loc[df.Prediction_from_prior_wk.notnull(), "Lunch_total"]
    dates_train = df.loc[df.Prediction_from_prior_wk.isnull(), "Date"]
    dates_test = df.loc[df.Prediction_from_prior_wk.notnull(), "Date"] 
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test

X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts()
```

<font size="4">Before we run our own predictions, let's see the human predictions and results:


```python
human_pred = lunch.loc[lunch.Prediction_from_prior_wk.notnull(), "Prediction_from_prior_wk"]

fig, ax = plt.subplots()
ax.plot(dates_test, human_pred, lw=4, color="tab:green", alpha=0.5, label="Human predictions")
ax.plot(dates_test, y_test, lw=4, color="tab:gray", alpha=0.5, label="Actual attendance")
ax.set_title("Human Predictions vs. Actual Lunch Attendance", fontsize=14)
ax.set_ylabel("Attendance (people)", fontsize=11)
ax.tick_params(axis="both", which="both", length=0)
ax.legend(fontsize=12)
fig.set_size_inches(8, 6)
plt.show()

```


![png](output_45_0.png)


<font size="4">Let's just look at the errors from each human prediction in relation to the Actual attendance:


```python
fig, ax = plt.subplots()
ax.plot(dates_test, human_pred-y_test, lw=4, color="tab:green", alpha=0.5, label="Human prediction Variance")
ax.plot(dates_test, y_test-y_test, lw=4, color="tab:gray", alpha=0.5, label="Actual attendance")
ax.set_title("Human Predictions vs. Actual Lunch Attendance: \nVariance", fontsize=14)
ax.set_ylabel("Prediction Variance (people)", fontsize=11)
ax.tick_params(axis="both", which="both", length=0)
ax.legend(fontsize=12)
fig.set_size_inches(8, 6)
plt.show()

```


![png](output_47_0.png)


<font size="4">Calculating out the MAE and RMSE, we see the human predictions had a MAE of 23.8 and a RMSE of 31.2. This means on average, the human prediction was off by 23.8 people. The first of the two numbers, 23.8 in this case, shows the MAE (Mean Absolute Error) metric we talked about earlier while the second number is the RMSE (Root Mean Squared Error).


```python
def calc_human_pred():
    human_MAE = metrics.mean_absolute_error(y_test, human_pred)
    human_RMSE = np.sqrt(metrics.mean_squared_error(y_test, human_pred))
    return human_MAE, human_RMSE

human_best = calc_human_pred()
print(calc_human_pred())
```

    (23.82758620689655, 31.192947920964443)
    

<font size="4">Now we'll make our first model. Let's see what happens if we simply use all the features (columns) in a linear regression model:


```python
def linreg_pred():
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_MAE = metrics.mean_absolute_error(y_test, y_pred)
    model_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return model, y_pred, model_MAE, model_RMSE, "linreg"

linreg_pred()[2:4]
```




    (34.123619782690476, 43.84771927945118)



<font size="4">This doesn't look too good. Using all the features, this model is off by 34 people on average and underperforms human predictions by 10 people.<br>
    
Let's choose some of the features most highly correlated with the Lunch attendance (both positively and negatively correlated) - **EN_prior_wk**, **Proximity_to_Major_Holiday**, **Is_Workday** and **Is_Holiday**.


```python
COLS = [
    'Is_Workday', 
    'Is_Holiday', 
#     'School_in_session',
    'Proximity_to_Major_Holiday', 
#     'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
#     'CN_prior_wk',
#     'Lunch_prior_wk', 
#     'EN_roll_2',
#     'CN_roll_2', 
#     'Lunch_roll_2', 
#     'EN_roll_4', 
#     'CN_roll_4', 
#     'Lunch_roll_4',
#     'EN_roll_8', 
#     'CN_roll_8', 
#     'Lunch_roll_8'
]

X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts(COLS=COLS)
linreg_results = linreg_pred()[2:4]
linreg_pred()[2:4]
```




    (26.24524813478226, 34.091120085200174)



<font size="4">This is a huge improvement, but the model still underperforms the human predictions by 2-3 people on average. Let's optimize a bit more by choosing some more features:


```python
COLS = [
    'Is_Workday', 
    'Is_Holiday', 
    'School_in_session',
#     'Proximity_to_Major_Holiday', 
#     'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
#     'CN_prior_wk',
    'Lunch_prior_wk', 
#     'EN_roll_2',
    'CN_roll_2', 
    'Lunch_roll_2', 
#     'EN_roll_4', 
    'CN_roll_4', 
#     'Lunch_roll_4',
    'EN_roll_8', 
#     'CN_roll_8', 
    'Lunch_roll_8'
]

X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts(COLS=COLS)
linreg_overfit = linreg_pred()[2:4]
linreg_pred()[2:4]
```




    (20.869899292965314, 27.019173910964852)



<font size="4">Now our MAE is 20.9, outperforming the human predictions by 3 people.<br>
    
But did you notice something strange about how we just improved the model from 26.2 to 20.9? When we only used 4 features that were highly correlated with Lunch attendance, we only got 26.2, but then after removing a highly correlated feature **Proximity_to_Major_Holiday** and adding in obscure and seemingly random rolling attendance features, we suddenly achieved 20.9. *Is this a quirk in the data or did the model pick up on some deeply nested thread of intelligence within these random features?*
    
    
    
    



<font size="4">Our model has *overfit* the data. Overfitting. The perennial problem that plagues predictive models.
    
>***Overfitting*** occurs when a predictive model learns not just the general trends or patterns in a dataset, but also the insignificant details that are not truly patterns in the data.
>
><font size="3">And this model overfit the data because there was a fundamental flaw in our methodology of optimizing our model. The flaw is that we are merely measuring our model's performance based on how well it did against a single metric - its performance against its human counterpart. By seeking to optimize against this metric, our model may pick up on *noise* within the dataset that happens to give it a better score on this metric, but the same model would perform weakly against brand new data.

<font size="4">So what's a better way to train a more robust model instead of a one-hit-wonder? We'll use a technique called *k-folds cross-validation*, which means that we'll take the training data, partition it into a certain number (*k*) of sets (*folds*), and then test the model on all combinations of these folds. Let's see how well these "26.2" and "20.9" models do when cross-validated:


```python
COLS = [
    'Is_Workday', 
    'Is_Holiday', 
#     'School_in_session',
    'Proximity_to_Major_Holiday', 
#     'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
#     'CN_prior_wk',
#     'Lunch_prior_wk', 
#     'EN_roll_2',
#     'CN_roll_2', 
#     'Lunch_roll_2', 
#     'EN_roll_4', 
#     'CN_roll_4', 
#     'Lunch_roll_4',
#     'EN_roll_8', 
#     'CN_roll_8', 
#     'Lunch_roll_8'
]

X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts(COLS=COLS)

linreg = LinearRegression()

print("MAE:", -cross_val_score(linreg, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean(), 
      "\nThe 5 Scores:", -cross_val_score(linreg, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"))
```

    MAE: 23.934993873233417 
    The 5 Scores: [20.59893043 29.44315001 16.42973437 24.44635992 28.75679463]
    

<font size="4">The first model scores 23.9 when cross-validated. We can see the distributions of the 5 individual scores based on the 5 *folds*. These scores vary widely from 16.4 to 29.4, depending on the different partitions of data for training and testing sets.<br>
    
Now let's look at the cross-validation score of the "20.9" model:


```python
COLS = [
    'Is_Workday', 
    'Is_Holiday', 
    'School_in_session',
#     'Proximity_to_Major_Holiday', 
#     'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
#     'CN_prior_wk',
    'Lunch_prior_wk', 
#     'EN_roll_2',
    'CN_roll_2', 
    'Lunch_roll_2', 
#     'EN_roll_4', 
    'CN_roll_4', 
#     'Lunch_roll_4',
    'EN_roll_8', 
#     'CN_roll_8', 
    'Lunch_roll_8'
]

X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts(COLS=COLS)
linreg = LinearRegression()
print("MAE:", -cross_val_score(linreg, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean(), 
      "\nThe 5 Scores:", -cross_val_score(linreg, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"))
```

    MAE: 34.474424402171316 
    The 5 Scores: [39.76252533 41.84884758 19.54826669 31.28336114 39.92912127]
    

<font size="4">While 1 of the 5 partitions scored a low 19.5, the other 4 partitions fared quite poorly. It turns out this bizare assortment of features was a cloud without rain.</font>

### *The moral of the story:*
<font size="4">Be careful not to "over-optimize" your model's performance against a specific data subset, such as performing well against the "human predictions" data subset. Doing so runs the risk of overfitting the model, making it much weaker to new data.


<font size="4">Let's move on to the second model we'll try:</font>

### "K-Nearest Neighbors" Regression

<font size="4">*K-Nearest Neighbors* makes predictions by finding the closest *k* number of data points ("*neighbors*") and averaging those for a prediction. 


```python
COLS = [
    'Is_Workday', 
    'Is_Holiday', 
    'School_in_session',
    'Proximity_to_Major_Holiday', 
    'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
    'CN_prior_wk',
    'Lunch_prior_wk', 
    'EN_roll_2',
    'CN_roll_2', 
    'Lunch_roll_2', 
    'EN_roll_4', 
    'CN_roll_4', 
    'Lunch_roll_4',
    'EN_roll_8', 
    'CN_roll_8', 
    'Lunch_roll_8'
]
X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts(COLS=COLS)

def knnreg_pred(k=5, w="uniform", xtn=X_train, xts=X_test, ytn=y_train, yts=y_test):
    model = KNeighborsRegressor(n_neighbors=k, weights=w)
    model.fit(xtn, ytn)
    y_pred = model.predict(xts)
    model_MAE = metrics.mean_absolute_error(yts, y_pred)
    model_RMSE = np.sqrt(metrics.mean_squared_error(yts, y_pred))
    return model, y_pred, model_MAE, model_RMSE, "KNNreg"

knnreg_ptl_overfit = knnreg_pred()[2:4]
knnreg_pred()[2:4]
```




    (23.0, 28.217211670942234)



## 2. Optimizing KNN

### *k-value*
<font size="4">Using all the columns, KNN outperformed the "full feature" linear regression model as well as the human predictions by just a little bit. Adjusting the features themselves like we did with the linear regression model only yields minimal gains, but with KNN, we can also tweak the "k" number of nearest neighbors to optimize prediction results.<br>

<font size="4">So let's further tune the model by finding out which value for "k" produces the best results.


```python
# mpl.rcParams['figure.dpi'] = 120

def plot_knn_error(start=1, stop=25):
    error = []
    for x in range(start,stop+1):
        model, y_pred, MAE, RMSE, name = knnreg_pred(k=x)
        human_pred.columns = ["Lunch_total"]    # Rename the Series column so it can be concatenated easier below.
        model_pred_delta = abs(y_pred - y_test)    # Manually calculate the 'error'
        human_pred_delta = abs(human_pred - y_test)
        delta_df = pd.concat([model_pred_delta, human_pred_delta], axis=1)
        delta_df.columns = ["model_AE", "human_AE"]
        delta_df["model_SE"] = delta_df.model_AE.apply(lambda x: x**2)    # Square the error
        delta_df["human_SE"] = delta_df.human_AE.apply(lambda x: x**2)
        error.append(list(delta_df.mean().apply(lambda x: x**.5 if x > 500 else x)))    
                        # Take the mean now of each observation in these 4 rows, and append this as a list to the 'error' list.
                        # For the two means which were Squared Errors, time to bring them back by taking the square root (x**.5)
    error_df = pd.DataFrame(error, columns=["model_AE", "human_AE", "model_SE", "human_SE"])
    error_df["k_value"] = error_df.index.values + 1
        
    fig, ax = plt.subplots()
    ax.plot(error_df.k_value, error_df.model_AE, "-o", linewidth=2, color="tab:green", alpha=0.7, label="Mean Absolute Error")
    ax.plot(error_df.k_value, error_df.model_SE, "-o", linewidth=2, color="tab:blue", alpha=0.7, label="Root Mean Squared Error")
    ax.plot(error_df.k_value, error_df.human_AE, "-o", linewidth=2, color="tab:green", alpha=0.2, label="Human MAE")
    ax.plot(error_df.k_value, error_df.human_SE, "-o", linewidth=2, color="tab:blue", alpha=0.2, label="Human RMSE")
    ax.set_ylabel("Average Error in Prediction \n(people)", alpha=0.8)
    ax.set_xlabel("k-Nearest Neighbor Value", alpha=0.8)
    ax.legend(fontsize=8)
    ax.set_title("K-Value Performance Comparison", fontsize=14, alpha=0.7)
    ax.tick_params(axis="both", which="both", length=0)
    fig.set_size_inches(8, 6)
    plt.show()
    
plot_knn_error(1, 25)
```


![png](output_66_0.png)


<font size="4">We see that with this feature selection, *k=5* optimizes the results.</font>

### *Feature Standardization*

<font size="4">Another way to improve model performance is by *scaling* the data.<br>

Since KNN calculates its estimate based on the distances between features, it's quite sensitive to different scales of values the features have. For instance, **EN_prior_wk** ranges from 67 to 317 people, whereas **Is_Workday** is simply a binary *True* or *False* (1 or 0) value. This means that the KNN model will not extract the full value of the data unless we scale the data to make the distances and relationships between the data points more meaningful.


```python
lunch.loc[:, ["EN_prior_wk", "Is_Workday"]].describe().loc[["min", "max"], :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EN_prior_wk</th>
      <th>Is_Workday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>67.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>317.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



<font size="4">We'll standardize the features and scale them within 0 and 1 and rerun the KNN model using the three most correlated features.<br>
    
We'll also perform cross-validation as we did for linear regression to make sure we're choosing the most robust model, not simply one that would "teach to the test". We will combine cross-validation with finding the best k-value by using the tool GridSearchCV.


```python
COLS = [
    'Is_Workday', 
#     'Is_Holiday', 
#     'School_in_session',
    'Proximity_to_Major_Holiday', 
#     'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
#     'CN_prior_wk',
#     'Lunch_prior_wk', 
#     'EN_roll_2',
#     'CN_roll_2', 
#     'Lunch_roll_2', 
#     'EN_roll_4', 
#     'CN_roll_4', 
#     'Lunch_roll_4',
#     'EN_roll_8', 
#     'CN_roll_8', 
#     'Lunch_roll_8'
]
X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts(COLS=COLS)

```

### *Without scaling*


```python
knn = KNeighborsRegressor()
params = {"n_neighbors":range(1,21)}
model = GridSearchCV(knn, params, cv=5, iid=True, scoring="neg_mean_absolute_error")
model.fit(X_train, y_train)
print(model.best_params_, model.best_score_, model.cv_results_['mean_test_score'], sep="\n")
```

    {'n_neighbors': 13}
    -31.63141025641026
    [-50.60416667 -43.04166667 -39.17361111 -35.70833333 -33.8
     -33.63194444 -34.20238095 -33.39583333 -33.30324074 -32.32916667
     -32.35037879 -31.75868056 -31.63141026 -32.21428571 -31.8
     -31.78645833 -31.89583333 -32.29398148 -32.00219298 -32.19166667]
    


```python
best_model = model.best_estimator_
y_pred = best_model.predict(X_test)
model_MAE = metrics.mean_absolute_error(y_test, y_pred)
model_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
knn_noscaling = (model_MAE, model_RMSE)
print(model_MAE, model_RMSE)
```

    26.310344827586203 32.54365956060272
    

<font size="4">Without scaling the data, our best KNN model achieved a 26.3. Now let's run the same experiment after scaling the data.

### *With scaling*


```python
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor()
params = {"n_neighbors":range(1,21)}
model = GridSearchCV(knn, params, cv=5, iid=True, scoring="neg_mean_absolute_error")
model.fit(X_train_scaled, y_train)
print(model.best_params_, model.best_score_, model.cv_results_['mean_test_score'], sep="\n")

```

    {'n_neighbors': 3}
    -26.57638888888889
    [-36.14583333 -30.91666667 -26.57638889 -27.84895833 -28.71666667
     -29.375      -28.0922619  -29.05989583 -29.06712963 -29.21041667
     -29.02462121 -29.296875   -28.82852564 -28.91220238 -29.925
     -29.80598958 -30.30269608 -30.23958333 -30.45614035 -30.66979167]
    


```python
best_model = model.best_estimator_
y_pred = best_model.predict(X_test_scaled)
model_MAE = metrics.mean_absolute_error(y_test, y_pred)
model_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
knn_scaling = (model_MAE, model_RMSE)
print(model_MAE, model_RMSE)
```

    24.482758620689655 29.052073738474284
    

<font size="4">Scaling the data yielded a 2-person increase in accuracy.

## 3. Optimize a Random Forest with GridSearchCV
<font size="4">Like we saw above, we could write functions to find and visualize the optimal parameters for a given model, but we don't need to since GridSearchCV does this for us. GridSearchCV tests a certain model with the range of given parameters that we input, determines the best model and allows us to then use that model for predictions.<br>
    
Let's make a random forest first, then find the optimal tuning for 3 parameters with GridSearchCV.


```python
COLS = [
    'Is_Workday', 
    'Is_Holiday', 
#     'School_in_session',
#     'Proximity_to_Major_Holiday', 
#     'Proximity_to_Minor_Holiday',
    'EN_prior_wk', 
#     'CN_prior_wk',
#     'Lunch_prior_wk', 
    'EN_roll_2',
#     'CN_roll_2', 
#     'Lunch_roll_2', 
#     'EN_roll_4', 
#     'CN_roll_4', 
    'Lunch_roll_4',
    'EN_roll_8', 
#     'CN_roll_8', 
#     'Lunch_roll_8'
]

X_train, X_test, y_train, y_test, dates_train, dates_test = lunch_tts(COLS=COLS)

def rf_pred():
    rf = RandomForestRegressor(random_state=10, n_estimators=50, min_samples_split=2, max_features=None,
                                 min_samples_leaf=5, max_depth=5)    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return rf, y_pred, MAE, RMSE, "forest"

rf, y_pred, MAE, RMSE, name = rf_pred()
print(MAE, RMSE)
```

    25.376831657299203 30.926707795369033
    

<font size="4">Our first Random Forest achieves an MAE of 25.4.<br>
    
Now we'll create a random forest using every possible combination of **max_depth** from 1-4, **min_samples_split** from 2-8, and **min_samples_leaf** from 1-3. This will produce 84 random forests, each having 50 decision trees.


```python
rf = RandomForestRegressor(random_state=10, n_estimators=50)
param_grid = {"max_depth":range(1,5), "min_samples_split":range(2,9), "min_samples_leaf":range(1, 4)}
grid = GridSearchCV(rf, param_grid, cv=5, iid=True, scoring="neg_mean_absolute_error")
grid.fit(X_train, y_train)

forest = grid.best_estimator_
y_pred = forest.predict(X_test)
MAE = metrics.mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
rf_results = (MAE, RMSE)
print("MAE: {}".format(MAE), "RMSE: {}".format(RMSE), "Best params:", grid.best_params_, sep="\n")
```

    MAE: 21.58475403898003
    RMSE: 27.012884006942137
    Best params:
    {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
    

<font size="4">Our fine-tuned forest now achieves 21.6 MAE. Not only did we optimize the tuning parameters, but by using GridSearchCV the model has been cross-validated already, which means it's optimized for any newly seen data and not simply fine-tuned to go head-to-head with the human predictions.

## 4. Sanity check our results with *dummy* models
<font size="4">We've run 3 models: linear regression, k-nearest neighbors and random forests. In many cases we won't be so fortunate to have human predictions against which to benchmark our models' predictive performances. A good way to check if our results are meaningful or not is by comparing our predictive model's performance against "non-intelligent" strategies, such as just guessing the average lunch attendance number every time.<br>

Let's run a dummy model that predicts the mean value every time.


```python
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
dummy_MAE = metrics.mean_absolute_error(y_test, y_pred)
dummy_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
dummy_results = (dummy_MAE, dummy_RMSE)
print(dummy_MAE, dummy_RMSE)
```

    35.19324712643678 41.84934326520462
    

<font size="4">So we see that a completely non-intelligent "dummy model" would be 35 people off each guess on average. This means any model that couldn't achieve better than this would be less than useless.

## 5. Final Results

<font size="4">Let's summarize the results from each of the models we predicted:


```python
final_results = [human_best, linreg_results, linreg_overfit, knnreg_ptl_overfit, 
                 knn_noscaling, knn_scaling, rf_results, dummy_results]
MAE_list = [a for a, b in final_results]
RMSE_list = [b for a, b in final_results]
names = ['Human Prediction', 'Linear Regression', 'Linear Regression \n(Overfit)', 'KNN \n(potential Overfit)',
        'KNN - un-scaled', 'KNN - scaled', 'Random Forest', '"Dummy" Model']
colors = ["tab:gray", "tab:blue", "tab:red", "tab:red", "tab:blue", "tab:blue", "tab:blue", "tab:gray"]
results_df = pd.DataFrame({'name':names, 'MAE':MAE_list, 'RMSE':RMSE_list, 'colors':colors})
results_df = results_df.sort_values("MAE", ascending=True)
```


```python
import matplotlib.patches as mpatch
fig, ax = plt.subplots()
ax.barh(results_df.name, results_df.RMSE, color=results_df.colors, alpha=0.2)
ax.barh(results_df.name, results_df.MAE, color=results_df.colors, alpha=0.6)
ax.invert_yaxis()
ax.set_title("Model Performances vs. Human & Dummy Benchmarks \n(MAE & RMSE scores)", fontsize=16)
ax.tick_params(axis="both", which="both", length=0)
ax.yaxis.set_tick_params(pad=10, labelsize=12)
for s in ["top", "bottom", "right", "left"]:
    ax.spines[s].set_color("white")
patch1 = mpatch.Patch(color="tab:red", label="Overfit model")
patch2 = mpatch.Patch(color="tab:blue", label="Robust model")
patch3 = mpatch.Patch(color="tab:gray", label="Benchmark")
ax.legend(handles=[patch2, patch3, patch1], fontsize=12)
fig.set_size_inches(10, 7)
plt.show()


```


![png](output_88_0.png)


<font size="4">These results show many things. First, the Human Prediction was much more accurate than simply a "dummy" guess based on a historical average. Also, several of our models outperformed the human predictions, but two of these either overfit the data or were at risk of overfitting; such models would be of little use to us in the real world. The Random Forest regression model outperformed human estimations, and also we see that k-nearest neighbors with scaled data performed better than using the raw, un-scaled data for KNN.

## Let's use our best model to make a prediction

<font size="4">Using the highest-performing *random forest* model, we can simply input the known data for a given week and get the model's prediction for lunch attendance.


```python
# We can change these variables to make each week's prediction
Is_Workday = 0    # 0 means not a workday
Is_Holiday = 0    # 0 means not a holiday
EN_prior_wk = 190
EN_roll_2 = 205
Lunch_roll_4 = 215
EN_roll_8 = 220

new_X = np.array([Is_Workday, Is_Holiday, EN_prior_wk, EN_roll_2,
                     Lunch_roll_4, EN_roll_8]).reshape(1, 6)
new_pred = forest.predict(new_X)
print("Lunch Attendance Prediction:", int(new_pred))
```

    Lunch Attendance Prediction: 195
    

# In Closing...
<font size="4">The human predictions are actually pretty good. With Lunch Attendance having a standard deviation of 52 people, estimating within 25 people on average is no simple feat.<br>
    
But while we saw there is room to improve upon with machine learning models, we also observed that the models that performed the best against the human predictions turned out to be overfit against the rest of the dataset. Using this kind of overfit model to predict against new data would lead to poor results. Like a teacher that may "teach to the test" only, pounding away at a machine learning model to optimize against a subset of the data (like the human predictions in our case) will just allow the model to prepare for that test instead of actually learning the trends and patterns within the dataset. That's why that red Linear Regression model dropped from a 20.8 to 34.5 MAE when we took it away from that data subset and subjected it to other data subsets via k-folds cross-validation. This poor score barely defeated the dummy model!

<font size="4">So yes, we can build models that outperform the human predictions, but ironically those best models would not be as robust to new real-world data as models that are trained with the goal of generalizing to new data rather than simply the goal of beating human predictions.



[//]: # (This is for displaying in GitHub Pages)

![]<img src="images/gz_catering_image.jpg?raw=true"/>



###### Final Notes

<font size="2"> 
    
1. The method we used to split up the final data into *train* and *test* sets was geared toward aligning our model's predictions with the 29 human predictions so that we could have a somewhat apples-to-apples comparison. A limitation to using this method of splitting is that because the rows with the human estimates happened to be chronologically in front of most of the other training data, this model wouldn't have been able to be used when the actual human predictions were made unless we decided to use the data before 2016 (project for another time). However, having pointed this out...
2. Predictions on future data would not only not have this problem (since they would be on future data), but would be even more accurate than the models we made since it would have the additional *testing* data to learn from also, allowing our model to better learn the patterns in the data.
3. Also, had this model been used over the course of those 29 weeks, after each week's prediction and subsequent actual results, the model could be retrained on the updated data and thus make more accurate predictions (in the same way the human predictor was learning from the week-by-week data to adjust predictions accordingly). Perhaps these scenarios will be projects for another time.


*Catering image by <a href="https://pixabay.com/users/kaicho20-86142/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3819922">Ira Lee Nesbitt</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3819922">Pixabay</a>*
