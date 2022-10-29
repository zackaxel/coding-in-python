

import pandas as pd 
from prophet import Prophet 
from IPython.display import clear_output
import os 
import logging 

logging.getLogger('prophet').setLevel(logging.WARNING)

data_import = pd.read_csv("VOO.csv", parse_dates=["Date"])
data_import 

day_mapper = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}
data_import["DayOfWeek"] = data_import["Date"].map(lambda x: day_mapper[x.dayofweek])
data_import 

data_import["diff_from_previous_day"] = (data_import["Open"].diff()/ data_import["Open"]) * 100
data_import

data_import.groupby("DayOfWeek")["diff_from_previous_day"].mean()

dates = pd.date_range(start=data_import["Date"].min(), end=data_import["Date"].max())

date_table = pd.DataFrame(data={"Calendar Date:dates"})
date_table["Weekday"] = date_table["Calendar Date"].map(lambda x: day_mapper[x.dayofweek])
date_table 

data_import.to_clipboard()

full_calendar = pd.merge(left=date_table, right=data_import, how='left', left_on = 'Calendar Date', right_on = "Date")
full_calendar = full_calendar[~full_calendar["Weekday"].isin(["Saturday", "Sunday"])]
full_calendar.rename(columns={"Date : "Trading Day"}, inplace=True)
full_calendar

full_calendar.to_clipboard()

full_calendar = full_calendar.bfill(axis='rows').reset_index(drop=True)
full_calendar.to_clipboard()

full_calendar["Weekday"].value_counts()

def day_backtester(day, amount_to_invest, data): 
    temp_data = data[data["Weekday"] == day]
    temp_data["Shares Owned"] = amount_to_invest / temp_data["Open"]
    final_price = temp_data['Open'].iloc[-1]
    final_amount = temp_data["Shares Owned"].sum() * final_price 
    formatted_final_amount = "${:, .2f}".format(final_amount)
    return formatted_final_amount
    
    for i in full_calendar["Weekday"].unique(): 
        print(i, day_backtester(i, 750, full_calendar))
        
    def prophet_predictor(dataset):
        m = Prophet()
        m.fit(dataset)
        future = m.make_future_dataframe(periods = 7)
        forecast = m.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        
        forecast_data = []
        
        for i in full_calendar[full_calendar["Weekday"] == "Monday"]["Calendar Date"].iloc[402:].index:
            days_to_include = i - 15
            temp = full_calendar.iloc[days_to_include: i, [0, 3]]
            temp.rename(columns={"Calendar Date": "ds", "Open": "y"}, inplace = True)
            prophet_predictor(temp).to_pickle(f"three_week_forecast//{i}.pkl")
            clear_output(wait = True)
            print(i, "/", full_calendar[full_calendar["Weekday"]=="Monday"]["Calendar Date"].iloc[17:].index.max())
            
            files_to_append = []
            files_to_read = ["three_week_forecast//" + i for i in os.listdir("three_week_forecast")]
            
            for i in files_to_read: 
                files_to_append.append(pd.read_pickle(i))
                
            predicted_values = pd.concat(files_to_append).sort_values("ds")
            predicted_values.rename(columns={"ds": "Predicted_Date", "yhat": "Prediction"}, inplace=True)
            predicted_values
            
            
    
