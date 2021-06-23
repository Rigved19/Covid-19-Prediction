# Covid-19-Prediction
Given a dataset on the number of cases and deaths due to COVID-19, it's useful to know how the situation will progress in an area based on tthe current stats, so we can lock down, or loosen restrictions as needed.

Using the dataset : Continent and Country, Data, along with the day's cases and deaths and other important features like **"population density"** , **"diabetes prevelance"** , **"median age".**

The layers in the model are build from scratch using only Python libraries for better understanding the math behind concepts involved deep neural networks.
Matplotlib and Pandas are used for Data Visualisation

**Given below is the architecture of the model deployed in the multi - layer perceptron**
![Covid 19 Prediction](https://user-images.githubusercontent.com/63362412/123039521-efefcc00-d40f-11eb-9c78-b87f3004838a.PNG)

# Technical Aspect

## The project is divided in 2 parts
### 1. Data Cleaning and Data Preprcossing
#### (i) Data Cleaning
 * Taking care of null Values in different ways depending on the feauture column
 * Converting Dates to count of Days 
#### (ii) Data Preprcoessing and Feature Scaling
  * Label and Target Encoding
  * Normalization
  * Standardization
  * Sorting
  
 ### 2. Building Encoder-Decoder Model
#### * During Training
  * Following are the function being implemented :
    * Initialize Properties
    * Parameters Initialization
    * Net1 Forward Propogation
    * Concatenation of other features + forward prop of Net1 = Net2
    * Net2 Forward Propogation
    * Decoder Forward Propogation
    * Loss - Mean Square Error
    * Net2 Backward Propogation
    * Net1 Backward Propogation
    * Gradient Descent

### * During Testing
   **Same steps as above, notable difference is prdiction for a given day is built on the prediction of all the prev days recursively while making prediction for next day**
 ### 3. Activation Functions :
     * Net1 to Net2 - Linear
     * Hidden Layers - ReLu
     * Output Layer - Linear

