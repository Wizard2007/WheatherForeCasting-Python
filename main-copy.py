# This is a sample Python script.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.M = hidden_dim
        self.L = layer_dim

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True)
        # batch_first to have (batch_dim, seq_dim, feature_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
      # initial hidden state and cell state
      h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
      c0 = torch.zeros(self.L, X.size(0), self.M).to(device)
  
      out, (hn, cn) = self.rnn(X, (h0.detach(), c0.detach()))
  
      # h(T) at the final time step
      out = self.fc(out[:, -1, :])
      return out

# Training
def train(model,
          learning_rate,
          X_train,
          y_train,
          X_test,
          y_test,
          epochs=200):

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Train loss
        train_losses[epoch] = loss.item()

        # Test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[epoch] = test_loss.item()

        if (epoch + 1) % 50 == 0:
            print(
                f'At epoch {epoch + 1} of {epochs}, Train Loss: {loss.item():.3f}, Test Loss: {test_loss.item():.3f}')

    return train_losses, test_losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def forward(self, X):
    # initial hidden state and cell state
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
    c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

    out, (hn, cn) = self.rnn(X, (h0.detach(), c0.detach()))

    # h(T) at the final time step
    out = self.fc(out[:, -1, :])
    return out

print(device)

weather = pd.read_csv("Istanbul,Turkey.csv",
                      on_bad_lines='skip')
weather = weather[
    ['datetime', 'temp', 'dew', 'humidity', 'precip', 'preciptype', 'windspeed', 'winddir',
     'cloudcover']]

weather = weather.rename(columns={'temp': 'temperature', 'datetime': 'date'})
weather['date'] = pd.to_datetime(weather['date'])
weather.set_index('date')
weather.head(5)

weather['preciptype'] = weather['preciptype'].fillna('None')
weather['cloudcover'] = weather['cloudcover'].fillna(0)
weather.isna().sum()

weather.dropna(inplace=True)
weather.info

weather['month'] = weather['date'].dt.month
weather.head(2)

weather_seperated = weather[['date', 'month', 'temperature']].copy(deep=True)
weather_seperated['year'] = weather['date'].dt.year
weather_seperated.drop(['date'], axis=1, inplace=True)
weather_seperated = weather_seperated.pivot_table(index='month', columns='year', values='temperature')
weather_seperated.head(2)

weather_seperated.plot()
plt.ylabel('Temperature (degrees Celcius)')
plt.title("Istanbul's Monthly Temperature Averages, Each Line Represents a Year")
plt.legend().remove()
plt.show()

g = sns.PairGrid(weather[['temperature', 'dew', 'humidity', 'windspeed']])
g.map(sns.scatterplot)

weather_naive = weather[['date', 'temperature']].copy(deep=True)
weather_naive['prev_temperature'] = weather_naive['temperature'].shift(1)
weather_naive.drop([0], inplace=True)
weather_naive['difference'] = weather_naive['temperature'] - weather_naive['prev_temperature']
weather_naive['square_error'] = weather_naive['difference'] ** 2
weather_naive.head(2)

square_error = weather_naive['square_error'].mean()
print(f'Square Error of the Naive Approach is {square_error:.3f}')
weather.head(2)

# One-hot-encoding precipitation type and month
weather_LSTM = weather.copy(deep=True)
weather_LSTM = pd.get_dummies(weather, columns=['preciptype', 'month'])
weather_LSTM.columns

weather_LSTM.head(2)

input_data = weather_LSTM.drop(['date'], axis=1)
targets = weather_LSTM['temperature'].values

T = 20  # Number of timesteps to look while predicting
D = input_data.shape[1]  # Dimensionality of the input
N = len(input_data) - T
print(f'Dimensions are {T} × {D} × {N}')

# Train size: 80% of the total data size
train_size = int(len(input_data) * 0.80)

# Normalization of the inputs
scaler = StandardScaler()
scaler.fit(input_data[:train_size + T - 1])
input_data = scaler.transform(input_data)

# Preparing X_train and y_train
X_train = np.zeros((train_size, T, D))
y_train = np.zeros((train_size, 1))

for t in range(train_size):
    X_train[t, :, :] = input_data[t:t + T]
    y_train[t] = (targets[t + T])

# Preparing X_test and y_test
X_test = np.zeros((N - train_size, T, D))
y_test = np.zeros((N - train_size, 1))

for i in range(N - train_size):
    t = i + train_size
    X_test[i, :, :] = input_data[t:t + T]
    y_test[i] = (targets[t + T])

# Make inputs and targets
X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

model = LSTM(D, 512, 2, 1)
model.to(device)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

train_losses, test_losses = train(model,
                                  0.01,
                                  X_train,
                                  y_train,
                                  X_test,
                                  y_test,
                                  epochs=750)

# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.xlabel('epoch no')
plt.ylabel('loss')
plt.legend()
plt.show()

# Checking one-step prediction performance of the model
test_target = y_test.cpu().detach().numpy()
test_predictions = []

for i in range(len(test_target)):
    input_ = X_test[i].reshape(1, T, D)
    p = model(input_)[0, 0].item()

    # update the predictions list
    test_predictions.append(p)

plot_len = len(test_predictions)
plot_df = weather[['date', 'temperature']].copy(deep=True)
plot_df = plot_df.iloc[-plot_len:]
plot_df['prediction'] = test_predictions
plot_df.set_index('date', inplace=True)
plot_df.head(5)

plt.plot(plot_df['temperature'], label='Actual Temperature', linewidth=1)
plt.plot(plot_df['prediction'], label='One-step Prediction', linewidth=1)
plt.xlabel('date')
plt.ylabel('temperature (degrees Celcius)')
plt.legend(loc='lower right')

LTSM_error = pd.DataFrame(test_target, columns=['targets'])
LTSM_error['predictions'] = test_predictions
LTSM_error['error'] = LTSM_error['targets'] - LTSM_error['predictions']
LTSM_error['error_square'] = LTSM_error['error'] ** 2
err = LTSM_error['error_square'].mean()
print(f'Mean square error is: {err:.3f}')

plt.hist(LTSM_error['error'], bins=25)
plt.xlabel('Temperature Difference (real - predictied)')
plt.ylabel('count')
plt.title('Distribution of Differences')



