# Big-Mart-Sales-Prediction-
🛒 Big Mart Sales Prediction Project
📌 Project Overview
The Big Mart Sales Prediction project is a machine learning application designed to forecast the sales of products at various Big Mart outlets. The goal is to help the business understand how different factors — such as product features and outlet characteristics — impact the sales performance of an item.

Using historical data, this project builds a predictive model that can estimate the Item_Outlet_Sales for any given product based on its features.

🧠 What the Project Does
The model predicts the expected sales of an item based on inputs like:

Item Weight

Item Fat Content

Item Visibility

Item MRP

Item Type (encoded)

Outlet Identifier (encoded)

Outlet Size, Location Type, and Type

Outlet Establishment Year

Item Identifier (encoded)

It takes these inputs from a user via a Streamlit-based user interface and displays the predicted sales value.

🔧 Tools & Technologies Used
Python – Core programming language

Pandas – Data loading and preprocessing

NumPy – Numerical computations

Matplotlib / Seaborn – Data visualization

Scikit-learn – Data preprocessing and model evaluation

XGBoost – Regressor for high-performance prediction

Streamlit – To create an interactive web UI

Pickle – To save and load the trained model

🛠️ How It Was Built
Data Collection & Cleaning

Loaded dataset from CSV (Train.csv)

Handled missing values using mean and mode

Normalized inconsistent category names

Exploratory Data Analysis (EDA)

Visualized distributions and category counts using Seaborn

Gained insights into which factors may influence sales

Feature Engineering

Applied Label Encoding to convert categorical columns

Selected relevant features for training

Model Training

Used XGBRegressor to train the model

Evaluated performance using R² score on both training and test sets

Model Saving

Saved the trained model using pickle for future use

Streamlit UI

Built an intuitive web interface where users enter product and outlet details

Displays the predicted sales instantly after clicking the "Predict" button
