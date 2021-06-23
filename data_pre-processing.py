!pip install category_encoders
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt


df = pd.read_csv("/content/covid-data (3).csv") #Reading the Data


df.info() #Getting info about the data


#Considering only relvant features(Columns)
df= df[["continent", "location" , "date" , "new_cases" , "new_deaths" , "positive_rate" , "population_density" , "median_age" , "diabetes_prevalence" , "life_expectancy" ]]


df.isnull().sum()


#Data Preprocessing
df.dropna(subset = ["continent"] , inplace = True)  #Remove Null values corr to Coninent

df.drop(["positive_rate"] , axis = 1 , inplace = True)  # Drop Postive Rate Column
 #Remove Null values corr to following columns
df.dropna(subset = ["population_density" , "median_age" , "life_expectancy" ,  "diabetes_prevalence"] , inplace = True)

 #Apply before fill to New Cases and New Deaths
df["new_deaths"].fillna( method = "bfill" , inplace = True)
df["new_cases"].fillna( method = "bfill" , inplace = True)
df["continent"].astype("category")  #Only after changing it to category can we apply the encoding techniques
df["location"].astype("category")   #Only after changing it to category can we apply the encoding techniques

#Coverting date into Unifirm format
df["date"] = pd.to_datetime(df["date"])

# Converting dates into days
k = []
for x in df['date']:
  k.append((x-df["date"][0]).days)   #We have to convert into list and add seperatly into the Dataframe
                                     #since, we cant append directly into the dataframe, no method availbale like that
df["Day"] = k
df.info(210)


#Feature Encoding
le = LabelEncoder()
te = TargetEncoder()
df["Continent_encoded"] = le.fit_transform(df["continent"])  #LabelEncoding for Continents
df["Location_encoded"] = te.fit_transform( df["location"] , df["new_cases"]) #TargetEncoding for Countries wrt New_Cases


#Calculating Previus Days Cases and Deaths
prev_day_cases = []
prev_day_cases.append(0)
for x in range(1 , df.shape[0]):
  if (df.iloc[x]["Location_encoded"] == df.iloc[x-1]["Location_encoded"]) :    #Use .iloc not .loc, since index are also deleted after deleting the rows
    prev_day_cases.append(df.iloc[x-1]["new_cases"])                           #As soon as Location changes, counter starts back from 0 cases for prev cases

  else :
    prev_day_cases.append(0)
df["Prev_Day_Cases"] = prev_day_cases   

#Calculating Previus Days Cases and Deaths
prev_day_deaths = []
prev_day_deaths.append(0)
for x in range(1 , df.shape[0]):
  if (df.iloc[x]["Location_encoded"] == df.iloc[x-1]["Location_encoded"]) :    #Use .iloc not .loc, since index are also deleted after deleting the rows
    prev_day_deaths.append(df.iloc[x-1]["new_deaths"])                         #As soon as Location changes, counter starts back from 0 cases for prev cases

  else :
    prev_day_deaths.append(0)  

df["Prev_Day_Deaths"] = prev_day_deaths


df.sort_values( by = ["Day" , "location"] , ascending =[True , True] , inplace = True ) #Sorting in ascending order wrt Days then, Location (Aplabeticallyl)
df["location"].astype("category")  #Returns the no. of countries


#Train Test Data Splitting
train_data = df[df["Day"] <= 200]  #df[Day] >= 200 returns Boolean Values to the condiction when fed to the entire data, passes only the rows corres. to values 
test_data = df[df['Day']>200]
test_data = test_data[test_data['Day']<218]
  #Same Logic
train_data = train_data.sample(frac=1) #Randomly shuffles entire training dataset
train_data.head(100)


#Training Data into Input Features
#Input Layer 1  = 2 Features - Continent and Country (Location)
Xnet1_train = train_data[["Continent_encoded","Location_encoded"]] 

#Input Layer 2 = 7 features - Day , Prev Day Cases, Prev Day Deaths , Population Density , Median Age , Diabetes Prevelance , Life Expectaancy
Xnet2_train = train_data[["Day" , "Prev_Day_Cases" , "Prev_Day_Deaths" , "population_density" , "median_age" , "diabetes_prevalence" , "life_expectancy"]]

#Output Features (Target)
Ytrain = train_data[["new_cases" ,"new_deaths"]]

#Testing Data into Input Features
test_data=test_data[test_data['location']!='Hong Kong'] #Honk Kong has no data for new_cases, new_deaths
Xnet1_test = test_data[["Continent_encoded","Location_encoded"]] 
#Considering Days as a seperate entity for 
Xnet2_test = test_data[["Prev_Day_Cases" , "Prev_Day_Deaths" , "population_density" , "median_age" , "diabetes_prevalence" , "life_expectancy"]]
test_days = test_data["Day"].to_numpy()                                        #Conversion is importnat calculation can be done only with arrays not lists
Ytest = test_data[["new_cases" ,"new_deaths"]]

#Training Feature Dimensions
print("Shape of Input features of Network1 " +str(Xnet1_train.shape))
print("Shape of Input features of Network2 " + str(Xnet2_train.shape))
print("Shape of Target of Network " + str(Ytrain.shape))
print(test_days.shape)


#Feature Scaling 
MinMax = MinMaxScaler(feature_range=(-10,10))
Standard = StandardScaler()
#Processing the Traning Data
#Normalizing Data -(Location_ Encoded, Continent_Encoded, Population Density , Median Age , Diabetes Prevelance , Life Expectancy between (-10 , 10)
#Net 1
Xnet1_train_scaled= MinMax.fit_transform(Xnet1_train) 

#Net2
Xnet2_train_scaled = Xnet2_train
Xnet2_train_scaled["population_density"] = MinMax.fit_transform(Xnet2_train["population_density"].to_numpy().reshape(-1,1))  #Need to reshape into 2D
Xnet2_train_scaled["median_age"] = MinMax.fit_transform(Xnet2_train["median_age"].to_numpy().reshape(-1,1))
Xnet2_train_scaled["diabetes_prevalence"] = MinMax.fit_transform(Xnet2_train["diabetes_prevalence"].to_numpy().reshape(-1,1))
Xnet2_train_scaled["life_expectancy"] = MinMax.fit_transform(Xnet2_train["life_expectancy"].to_numpy().reshape(-1,1))

#Standardize Data -( Prev_Day_Cases , Prev_Day_Deaths , Target )
Xnet2_train_scaled["Prev_Day_Cases"] = Standard.fit_transform(Xnet2_train["Prev_Day_Cases"].to_numpy().reshape(-1,1)) 
Xnet2_train_scaled["Prev_Day_Deaths"] = Standard.fit_transform(Xnet2_train["Prev_Day_Deaths"].to_numpy().reshape(-1,1))
Ytrain_scaled = Standard.fit_transform(Ytrain)                          
Xnet2_train_scaled = Xnet2_train_scaled.to_numpy()    #Calculation can be done only on arrays and not list

#Processing the Testing Data
Xnet1_test_scaled = MinMax.fit_transform(Xnet1_test)

Xnet2_test_scaled = Xnet2_test
Xnet2_test_scaled["population_density"] = MinMax.fit_transform(Xnet2_test["population_density"].to_numpy().reshape(-1,1))  #Need to reshape into 2D
Xnet2_test_scaled["median_age"] = MinMax.fit_transform(Xnet2_test["median_age"].to_numpy().reshape(-1,1))
Xnet2_test_scaled["diabetes_prevalence"] = MinMax.fit_transform(Xnet2_test["diabetes_prevalence"].to_numpy().reshape(-1,1))
Xnet2_test_scaled["life_expectancy"] = MinMax.fit_transform(Xnet2_test["life_expectancy"].to_numpy().reshape(-1,1))

Xnet2_test_scaled["Prev_Day_Cases"] = Standard.fit_transform(Xnet2_test["Prev_Day_Cases"].to_numpy().reshape(-1,1)) 
Xnet2_test_scaled["Prev_Day_Deaths"] = Standard.fit_transform(Xnet2_test["Prev_Day_Deaths"].to_numpy().reshape(-1,1))
Xnet2_test_scaled = Xnet2_test_scaled.to_numpy()

Ytest_scaled = Ytest
Ytest_scaled["new_cases"] = Standard.fit_transform(Ytest["new_cases"].to_numpy().reshape(-1,1))
Ytest_scaled["new_deaths"] = Standard.fit_transform(Ytest["new_deaths"].to_numpy().reshape(-1,1))
Ytest_scaled= Ytest_scaled.to_numpy()

t = Xnet2_test_scaled.T
p = t[:2,180:360]
print(p.shape)
#DataType
#Xnet1_train_scaled = nd.array
#Xnet2_train_scaled = nd.array
#Ytrain_scaled = nd.array

#Xnet1_test_scaled = nd.array
#Xnet2_test_scaled = nd.array
#Ytestscaled = nd.array
print(type(Ytest_scaled))

