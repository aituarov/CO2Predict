import streamlit as st

from sklearn.linear_model import Lasso
# from sklearn import svm
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn import tree
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

from pandas.tseries.offsets import MonthEnd, MonthBegin
from datetime import datetime
import holidays

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import pickle


def read_file(file):
    employees_data = pd.read_excel(file, sheet_name='Employees')
    trips_data = pd.read_excel(file, sheet_name='Trips')
    return employees_data, trips_data

def transform_date_columns(trips_data):
    dw_mapping={
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    
    trips_data['TimeDuration'] = round((trips_data.StopTime - trips_data.StartTime) / np.timedelta64(1, 'h'), 2)
    trips_data['Date'] = trips_data.StartTime.dt.date
    trips_data['StartTime'] = trips_data.StartTime.dt.time
    trips_data['EndTime'] = trips_data.StopTime.dt.time
    trips_data.drop(columns='StopTime', inplace=True)
    trips_data['Date'] = pd.to_datetime(trips_data['Date'])
    trips_data['Day_of_week'] = trips_data.Date.dt.weekday.map(dw_mapping)
    return trips_data

# def get_stat_by_group(group_column, stat_columns, df):
#     group_names = df[group_column].unique()
#     info = pd.DataFrame(columns=['group', 'Q1', 'mean', 'median', 'Q3', 'IQR', 'count', 'min_val', 'max_val'])
#     for g in group_names:
#         data = df.loc[df[group_column] == g]
#         q1 = data[stat_columns].quantile(0.25)
#         mean = data[stat_columns].mean()
#         median = data[stat_columns].quantile(0.5)
#         q3 = data[stat_columns].quantile(0.75)
#         iqr = q3 - q1
#         min_val = q1 - 1.5 * iqr
#         max_val = q3 + 1.5 * iqr
#         info.loc[len(info)] = [g, q1, mean, median, q3, iqr, len(data), min_val, max_val]
#     return info

def delete_outliers(commute_trips):
    commute_trips = commute_trips.loc[commute_trips['Co2_emissions'] > 0]
    commute_trips = commute_trips.loc[commute_trips['TimeDuration'] < 4]
    commute_trips = commute_trips[~commute_trips.Day_of_week.isin(['Saturday', 'Sunday'])]

    commute_trips.loc[(commute_trips['FuelType'].notnull()) &
               (commute_trips['Vehicle'].isin(['Bike', 'Foot', 'SharedBike'])), 'FuelType'] = np.NaN
    commute_trips['VehicleWithFuel'] = np.where(
        (~commute_trips['FuelType'].isnull()) & (commute_trips['Vehicle'].str.contains('Car')),
        commute_trips['Vehicle'][(~commute_trips['FuelType'].isnull()) &
        (commute_trips['Vehicle'].str.contains('Car'))] + ' ' + commute_trips['FuelType'], commute_trips['Vehicle'])
    commute_trips = commute_trips.loc[~commute_trips['Vehicle'].isin(['Plane', 'Boat', 'SharedElectricCar', 'Motorbike'])]
    commute_trips['Vehicle'] = np.where((commute_trips['FuelType'] == 'electrisch') &
     (commute_trips['Vehicle'].str.contains('Car')), 'ElectricCar', commute_trips['Vehicle'])

    commute_trips['Avg Velocity'] = commute_trips['Distance'] // commute_trips['TimeDuration']
    max_avg_velocity= {'Car': 120, 'Bike': 25, 'Foot': 15, 'PublicTransport': 70,
                       'Carpool': 120, 'ElectricBike': 25, 'Train': 150,
                       'ElectricCar': 120, 'SharedBike': 25, 'Moped': 50}

    for i in commute_trips['Vehicle'].values:
      outliers_by_velocity = pd.DataFrame(columns=commute_trips.columns)
      human_max_velocity = max_avg_velocity[i]
      data = commute_trips.loc[(commute_trips['Vehicle'] == i) & (commute_trips['Avg Velocity'] > human_max_velocity)]
      outliers_by_velocity = pd.concat([outliers_by_velocity, data])
    commute_trips = commute_trips.drop(outliers_by_velocity.index)
    return commute_trips
    
def lag_feature(df_grouped, shift_months, cols):
  for c in cols:
    tmp = df_grouped[['Date', 'CompanyId', 'VehicleWithFuel', c]]
    for i in range(1, shift_months+1):
      shifted = tmp.copy()
      shifted.columns = ['Date', 'CompanyId', 'VehicleWithFuel', c + "_lag_"+str(i)]
      shifted.Date = shifted.Date + pd.DateOffset(months=i)
      df_grouped = pd.merge(df_grouped, shifted,
                            on=['Date', 'CompanyId', 'VehicleWithFuel'], how='left')
  return df_grouped

def count_workdays(df):
    h = holidays.Netherlands()
    b = pd.bdate_range(df['Date'], df['Date'] + MonthEnd(0))
    return sum(y not in h for y in b)

def iterative_predict(model, X, y_feature, batch_size, cnt_predict):
  iters = range(int(len(X)/batch_size))
  y_pred = None
  predictions = []

  for _ in iters:
    index = [_ * batch_size + i for i in range(batch_size)]
    X_cur = X.iloc[index]

    if y_pred is not None:
      for i, f in enumerate(y_feature):
        if _ >= 1:
          if _ == 2:
            X_cur[f'{f}_2'] = predictions[0]
          
          if len(y_feature) == 1:
            X_cur[f'{f}_1'] = predictions[-1]
          else:
            X_cur[f'{f}_1'] = predictions[0][:, i]

          if cnt_predict:
            X_cur['qmean_cnt'] = X_cur[['Count_lag_1',
                                    'Count_lag_2',
                                    'Count_lag_3']].mean(skipna=True, axis=1)
            # Add quater std count
            X_cur['qstd_cnt'] = X_cur[['Count_lag_1',
                                                'Count_lag_2',
                                                'Count_lag_3']].std(skipna=True, axis=1)
            # Add quater min count
            X_cur['qmin_cnt'] = X_cur[['Count_lag_1',
                                                'Count_lag_2',
                                                'Count_lag_3']].min(skipna=True, axis=1)
            # Add quater max count
            X_cur['qmax_cnt'] = X_cur[['Count_lag_1',
                                                'Count_lag_2',
                                                'Count_lag_3']].max(skipna=True, axis=1)
  
    y_pred = model.predict(X_cur)
    # predictions = pd.concat([predictions, y_pred])
    predictions.append(y_pred)
  return predictions

def plot_timeseries(actual_data, predicted_data, actual_col='Actual', predicted_col='Predicted', title=None, xlabel=None, ylabel=None):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual data
    ax.plot(actual_data.index, actual_data[actual_col], label='Actual', marker='o')

    # Plot predicted data
    ax.plot(predicted_data.index, predicted_data[predicted_col], label='Predicted', marker='o')

    for x, y in zip(predicted_data.index, predicted_data[predicted_col]):
        ax.annotate(f"{round(y)}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Show grid and legend
    ax.grid(True)
    ax.legend()

    # Format the date on the x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

    # Rotate the x-axis labels for better readability (optional)
    plt.xticks(rotation=45)

    # Display the plot
    plt.tight_layout()
    # plt.show()
    return fig


# def make_test_df(df, n_months):
#   # print(df)
#   last_month = df.Date.max()
#   add_dates = [last_month + pd.DateOffset(months=i) for i in range(1, n_months+1)]
#   all_values = pd.MultiIndex.from_product([add_dates, df['CompanyId'].unique(),
#                                            df['VehicleWithFuel'].unique()])
#   test_df = pd.DataFrame(index=all_values).reset_index()
#   test_df.columns = ['Date', 'CompanyId', 'VehicleWithFuel']
#   test_df[['TimeDuration_mean', 'Distance_mean', 'Co2_mean', 'Co2_sum', 'Count']] = 0

#   df_grouped = df.groupby(['Date', 'CompanyId', 'VehicleWithFuel'], as_index=False)
#   df_grouped = df_grouped.agg({'TimeDuration': ['mean'], 'Distance': ['mean'],
#                           'Co2_emissions': ['mean', 'sum', 'count']})

#   df_grouped.columns = ['Date', 'CompanyId', 'VehicleWithFuel',
#                 'TimeDuration_mean', 'Distance_mean', 'Co2_mean', 'Co2_sum', 'Count']
#   df_grouped= pd.concat([df_grouped, test_df], ignore_index=True)

#   # df_grouped = df.groupby(['Date', 'CompanyId', 'VehicleWithFuel'], as_index=False)
#   df_grouped_with_lag = lag_feature(df_grouped, 3,
#       ['TimeDuration_mean', 'Distance_mean', 'Co2_mean', 'Count'])

#   df_grouped_with_lag.fillna(0, inplace=True)
#   df_grouped_with_lag['qmean_cnt'] = df_grouped_with_lag[['Count_lag_1',
#                                   'Count_lag_2',
#                                   'Count_lag_3']].mean(skipna=True, axis=1)
#   # Add quater std count
#   df_grouped_with_lag['qstd_cnt'] = df_grouped_with_lag[['Count_lag_1',
#                                       'Count_lag_2',
#                                       'Count_lag_3']].std(skipna=True, axis=1)
#   # Add quater min count
#   df_grouped_with_lag['qmin_cnt'] = df_grouped_with_lag[['Count_lag_1',
#                                       'Count_lag_2',
#                                       'Count_lag_3']].min(skipna=True, axis=1)
#   # Add quater max count
#   df_grouped_with_lag['qmax_cnt'] = df_grouped_with_lag[['Count_lag_1',
#                                       'Count_lag_2',
#                                       'Count_lag_3']].max(skipna=True, axis=1)

#   ohe = OneHotEncoder()
#   le = LabelEncoder()
#   ohe_encode_columns = ['VehicleWithFuel']
#   le_encode_columns = ['CompanyId']
#   ohe_encoded_array = ohe.fit_transform(df_grouped_with_lag[ohe_encode_columns]).toarray()
#   ohe_encoded_labels = ohe.get_feature_names_out()
#   ohe_encoded_df = pd.DataFrame(ohe_encoded_array, columns=ohe_encoded_labels)
#   ohe_encoded_labels_list = ohe_encoded_labels.tolist()

#   trained_vehicle_list = pickle.load(open('vehicle_columns.sav', 'rb'))
#   fill_zero_columns = [i for i in trained_vehicle_list if i not in ohe_encoded_labels_list]

#   df_grouped_with_lag.drop(columns=ohe_encode_columns, inplace=True)
#   df_grouped_with_lag = pd.concat([df_grouped_with_lag, ohe_encoded_df], axis=1)
#   df_grouped_with_lag.dropna(inplace=True)
#   df_grouped_with_lag['CompanyId'] = le.fit_transform(df_grouped_with_lag[le_encode_columns])
#   df_grouped_with_lag[fill_zero_columns] = 0

#   nl_holidays = holidays.Netherlands()
#   months = df_grouped_with_lag.Date.unique()
#   df_grouped_with_lag['Nb_work_days'] =  df_grouped_with_lag.apply(count_workdays, axis=1)

#   count_columns = ['CompanyId', 'Count_lag_1', 'Count_lag_2', 'Count_lag_3',
#                   'qmean_cnt', 'qstd_cnt', 'qmin_cnt', 'qmax_cnt', 'Nb_work_days']
#   count_columns.extend(trained_vehicle_list)

#   # For co2_mean predict model
#   co2_columns = ['CompanyId',
#                   'TimeDuration_mean_lag_1', 'TimeDuration_mean_lag_2','TimeDuration_mean_lag_3',
#                   'Distance_mean_lag_1', 'Distance_mean_lag_2', 'Distance_mean_lag_3',
#                   'Co2_mean_lag_1', 'Co2_mean_lag_2','Co2_mean_lag_3']
#   co2_columns.extend(trained_vehicle_list)

#   df_grouped_with_lag.set_index(['Date'], inplace=True)

#   y_count = df_grouped_with_lag['Count']
#   X_count = df_grouped_with_lag.loc[:, count_columns]
#   y_co2 = df_grouped_with_lag[['TimeDuration_mean', 'Distance_mean', 'Co2_mean']]
#   X_co2 = df_grouped_with_lag.loc[:, co2_columns]
#   train_index = df_grouped_with_lag.index.isin([last_month])
#   predict_index = df_grouped_with_lag.index.isin(add_dates)

#   model_count = pickle.load(open('model_count.sav', 'rb'))
#   X_train, y_train, X_pred = X_count.loc[train_index, :], y_count[train_index], \
#   X_count.loc[predict_index, :]
#   model_count.fit(X_train, y_train)
#   count_predictions = iterative_predict(model_count, X_pred, ["Count_lag"], len(ohe_encoded_labels_list))
#   count_predictions = np.concatenate(count_predictions, axis=0)

#   model_co2_mean = pickle.load(open('model_co2_mean.sav', 'rb'))
#   X_train, y_train, X_pred = X_co2.loc[train_index, :], y_co2[train_index], \
#   X_co2.loc[predict_index, :]
#   model_co2_mean.fit(X_train, y_train)
#   mean_predictions = iterative_predict(model_co2_mean, X_pred, 
#    ['TimeDuration_mean_lag', 'Distance_mean_lag', 'Co2_mean_lag'], len(ohe_encoded_labels_list))
#   mean_predictions = np.concatenate(mean_predictions, axis=0)
  
#   co2_mean_predictions = [row[-1] for row in mean_predictions]
#   fix_model = pickle.load(open('fix_model.sav', 'rb'))
#   co2_sum = np.multiply(count_predictions, co2_mean_predictions)
#   # print(co2_sum)
#   X_pred = pd.concat([df_grouped_with_lag.loc[predict_index, trained_vehicle_list].reset_index(drop=True), pd.Series(co2_sum, name='Co2_sum')], axis=1)
#   X_pred.index = test_df.Date

#   results = fix_model.predict(X_pred)
#   input_df_agg = df[['Date', 'Co2_emissions']].set_index('Date').groupby('Date').sum()
#   input_df_agg['Type'] = 'Actual'
#   results_agg = pd.DataFrame(results, index=X_pred.index, columns=['Co2_emissions']).groupby('Date').sum()
#   results_agg['Type'] = 'Predicted'
    
#   return input_df_agg, results_agg

def make_test_df(df, n_months):
  # print(df)
  last_month = df.Date.max()
  add_dates = [last_month + pd.DateOffset(months=i) for i in range(1, n_months+1)]
  all_values = pd.MultiIndex.from_product([add_dates, df['CompanyId'].unique(),
                                           df['VehicleWithFuel'].unique()])
  test_df = pd.DataFrame(index=all_values).reset_index()
  test_df.columns = ['Date', 'CompanyId', 'VehicleWithFuel']
  test_df[['TimeDuration_mean', 'Distance_mean', 'Co2_mean', 'Co2_sum', 'Count']] = 0

  df_grouped = df.groupby(['Date', 'CompanyId', 'VehicleWithFuel'], as_index=False)
  df_grouped = df_grouped.agg({'TimeDuration': ['mean'], 'Distance': ['mean'],
                          'Co2_emissions': ['mean', 'sum', 'count']})

  df_grouped.columns = ['Date', 'CompanyId', 'VehicleWithFuel',
                'TimeDuration_mean', 'Distance_mean', 'Co2_mean', 'Co2_sum', 'Count']
  df_grouped= pd.concat([df_grouped, test_df], ignore_index=True)

  # df_grouped = df.groupby(['Date', 'CompanyId', 'VehicleWithFuel'], as_index=False)
  df_grouped_with_lag = lag_feature(df_grouped, 3,
      ['TimeDuration_mean', 'Distance_mean', 'Co2_mean', 'Count'])

  df_grouped_with_lag.fillna(0, inplace=True)
  df_grouped_with_lag['qmean_cnt'] = df_grouped_with_lag[['Count_lag_1',
                                  'Count_lag_2',
                                  'Count_lag_3']].mean(skipna=True, axis=1)
  # Add quater std count
  df_grouped_with_lag['qstd_cnt'] = df_grouped_with_lag[['Count_lag_1',
                                      'Count_lag_2',
                                      'Count_lag_3']].std(skipna=True, axis=1)
  # Add quater min count
  df_grouped_with_lag['qmin_cnt'] = df_grouped_with_lag[['Count_lag_1',
                                      'Count_lag_2',
                                      'Count_lag_3']].min(skipna=True, axis=1)
  # Add quater max count
  df_grouped_with_lag['qmax_cnt'] = df_grouped_with_lag[['Count_lag_1',
                                      'Count_lag_2',
                                      'Count_lag_3']].max(skipna=True, axis=1)

  ohe = OneHotEncoder()
  # le = LabelEncoder()
  ohe_encode_columns = ['VehicleWithFuel']
  # le_encode_columns = ['CompanyId']
  ohe_encoded_array = ohe.fit_transform(df_grouped_with_lag[ohe_encode_columns]).toarray()
  ohe_encoded_labels = ohe.get_feature_names_out()
  ohe_encoded_df = pd.DataFrame(ohe_encoded_array, columns=ohe_encoded_labels)
  ohe_encoded_labels_list = ohe_encoded_labels.tolist()

  trained_vehicle_list = pickle.load(open('vehicle_columns.sav', 'rb'))
  fill_zero_columns = [i for i in trained_vehicle_list if i not in ohe_encoded_labels_list]

  df_grouped_with_lag.drop(columns=ohe_encode_columns, inplace=True)
  df_grouped_with_lag = pd.concat([df_grouped_with_lag, ohe_encoded_df], axis=1)
  df_grouped_with_lag.dropna(inplace=True)
  # df_grouped_with_lag['CompanyId'] = le.fit_transform(df_grouped_with_lag[le_encode_columns])
  df_grouped_with_lag[fill_zero_columns] = 0

  nl_holidays = holidays.Netherlands()
  months = df_grouped_with_lag.Date.unique()
  df_grouped_with_lag['Nb_work_days'] =  df_grouped_with_lag.apply(count_workdays, axis=1)
    
  df_grouped_with_lag['Year'] = df_grouped_with_lag['Date'].dt.year
  df_grouped_with_lag['Month'] = df_grouped_with_lag['Date'].dt.month
  df_grouped_with_lag['Week'] = df_grouped_with_lag['Date'].dt.isocalendar().week

  count_columns = ['Year', 'Month', 'Week', 'Count_lag_1', 'Count_lag_2', 'Count_lag_3',
                  'qmean_cnt', 'qstd_cnt', 'qmin_cnt', 'qmax_cnt', 'Nb_work_days']
  count_columns.extend(trained_vehicle_list)

  # For co2_mean predict model
  co2_columns = ['Year', 'Month', 'Week', 
                  'TimeDuration_mean_lag_1', 'TimeDuration_mean_lag_2','TimeDuration_mean_lag_3',
                  'Distance_mean_lag_1', 'Distance_mean_lag_2', 'Distance_mean_lag_3',
                  'Co2_mean_lag_1', 'Co2_mean_lag_2','Co2_mean_lag_3']
  co2_columns.extend(trained_vehicle_list)

  df_grouped_with_lag.set_index(['Date'], inplace=True)

  y_count = df_grouped_with_lag['Count']
  X_count = df_grouped_with_lag.loc[:, count_columns]
  y_co2 = df_grouped_with_lag['Co2_mean']
  X_co2 = df_grouped_with_lag.loc[:, co2_columns]
  train_index = df_grouped_with_lag.index.isin([last_month])
  predict_index = df_grouped_with_lag.index.isin(add_dates)

  model_count = pickle.load(open('new_model_count.sav', 'rb'))
  # X_train, y_train, X_pred = X_count.loc[train_index, :], y_count[train_index], \
  X_pred = X_count.loc[predict_index, :]
  # model_count.fit(X_train, y_train)
  count_predictions = iterative_predict(model_count, X_pred, ["Count_lag"], len(ohe_encoded_labels_list), True)
  count_predictions = np.concatenate(count_predictions, axis=0)

  model_co2_mean = pickle.load(open('new_model_co2_mean.sav', 'rb'))
  # X_train, y_train, X_pred = X_co2.loc[train_index, :], y_co2[train_index], \
  X_pred = X_co2.loc[predict_index, :]
  # model_co2_mean.fit(X_train, y_train)
  mean_predictions = iterative_predict(model_co2_mean, X_pred, 
   ['Co2_mean_lag'], len(ohe_encoded_labels_list), False)
  co2_mean_predictions = np.concatenate(mean_predictions, axis=0)
  
  # co2_mean_predictions = [row[-1] for row in mean_predictions]
  fix_model = pickle.load(open('fix_model.sav', 'rb'))
  co2_sum = np.multiply(count_predictions, co2_mean_predictions)
  # print(co2_sum)
  X_pred = pd.concat([df_grouped_with_lag.loc[predict_index, trained_vehicle_list].reset_index(drop=True), pd.Series(co2_sum, name='Co2_sum')], axis=1)
  X_pred.index = test_df.Date

  results = fix_model.predict(X_pred)
  input_df_agg = df[['Date', 'Co2_emissions']].set_index('Date').groupby('Date').sum()
  input_df_agg['Type'] = 'Actual'
  results_agg = pd.DataFrame(results, index=X_pred.index, columns=['Co2_emissions']).groupby('Date').sum()
  results_agg['Type'] = 'Predicted'
    
  return input_df_agg, results_agg


def main() -> None:
    st.header("Green Mobility tool")
    st.subheader("Upload your test excel file to make predictions")
    uploaded_data = st.file_uploader("Drag and Drop or Click to Upload", type = ".xlsx", accept_multiple_files = False)
    if uploaded_data is not None:
        employees_data, trips_data = read_file(uploaded_data)
        trips_data = transform_date_columns(trips_data)
        trips_data = trips_data[['EmployeeId', 'TripId', 'TripType', 'Vehicle', 'Date', 'Day_of_week',
                         'StartTime', 'EndTime', 'TimeDuration', 'Distance', 'Co2_emissions']]
        trips_data = pd.merge(trips_data, employees_data[['EmployeeId', 'CompanyId', 'FuelType']],
                      on='EmployeeId', how='left')
        commute_trips = trips_data.loc[trips_data['TripType'] == 'commute']
        updated_commute_trips = delete_outliers(commute_trips)

        model_data = updated_commute_trips.copy()
        model_data.drop(columns=['EmployeeId', 'TripId', 'TripType', 'Vehicle', 'FuelType', 'Day_of_week', 'EndTime', 'Avg Velocity'], inplace=True)
        model_data['StartTime'] = model_data['StartTime'].apply(lambda x: round(x.hour + x.minute/60, 2))
        
        model_data['Date'] = model_data['Date'] + MonthEnd(0) - MonthBegin(1)
        
        model_data_monthly = model_data.copy()
        model_data_monthly.sort_values(['VehicleWithFuel', 'CompanyId', 'Date'], inplace=True)

        input_df, predict_df = make_test_df(model_data_monthly, 6)
        fig = plot_timeseries(input_df, predict_df, actual_col='Co2_emissions', predicted_col='Co2_emissions', title='CO2 Emissions Actual vs. Predicted', xlabel='Date', ylabel='CO2 Emissions')
        st.pyplot(fig)

if __name__ == "__main__":
    st.set_page_config(
        "Green Mobility tool",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()
