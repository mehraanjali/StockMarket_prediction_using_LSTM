# Stock Market Price Prediction Using LSTM

## Project Description:

This project focuses on leveraging Long Short-Term Memory (LSTM) neural networks to predict stock market prices. LSTM is a type of recurrent neural network (RNN) known for its ability to model sequential data effectively, making it a popular choice for time series forecasting, such as stock price prediction.

## Project Overview:

In this project, we build and train an LSTM-based neural network to predict the future closing prices of a specific stock (in this case, AAPL - Apple Inc.) based on historical stock price data. We'll also evaluate the model's performance using Root Mean Square Error (RMSE) as a performance metric.

## Key Steps in the Project:

Data Collection:
We fetch historical stock price data for AAPL from the Tiingo API using the pandas_datareader library.
The data is saved as a CSV file for further analysis.

Data Preprocessing:
We preprocess the data to make it suitable for training an LSTM model.
Scaling: We apply Min-Max scaling to normalize the data, as LSTMs are sensitive to the scale of input features.
Sequence Creation: We create sequences of data with a specified time step, which will be used for training the LSTM model.

LSTM Model Building:
We design a stacked LSTM model using the TensorFlow/Keras library.
The model architecture consists of multiple LSTM layers followed by a Dense layer for prediction.
We compile the model with the mean squared error (MSE) loss function and the Adam optimizer.

Model Training:
We split the data into training and testing sets.
The LSTM model is trained on the training data, and its performance is evaluated on the testing data.
Training progress and model summary are displayed.

Model Evaluation:
We evaluate the model's performance using RMSE for both the training and testing datasets.
RMSE provides an indication of how well the model predicts stock prices.

Visualization:
We create visualizations to show the model's predictions alongside the actual stock prices.
The graphs display the original data, training predictions, and test predictions.

Future Price Prediction:
After training the model, we demonstrate how to use it to make future stock price predictions.
We take a seed sequence of historical data and iteratively predict future prices for a specified number of days.

## Project Outcomes:

This project serves as a practical example of applying deep learning techniques, specifically LSTM neural networks, for stock price prediction. By following this project, users can understand the following:

How to collect and preprocess financial time series data.
Building and training LSTM models for time series forecasting.
Evaluating model performance using RMSE.
Visualizing model predictions alongside actual data.
Using the trained model to make future price predictions.
