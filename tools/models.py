import torch
import torch.nn as nn
import torch.optim as optim

class CNN1D(nn.Module):
    def __init__(self, input_channels=2, num_classes=6):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (225 // 4), 128)  # Adjust based on input size after pooling
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=2, lstm_hidden_size=256, num_classes=6, num_lstm_layers=1):
        super(CNN_LSTM, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # LSTM
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, num_layers=1, bidirectional=True)

        self.dropout = nn.Dropout(0.5)

        # Fully Connected Layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # CNN Feature Extraction
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        # Reshape for LSTM (batch_size, time_steps, features)
        x = x.permute(0, 2, 1)  # Change shape from (batch, channels, time) → (batch, time, channels)
        
        # LSTM
        x, _ = self.lstm(x)  # Output shape: (batch, time_steps, lstm_hidden_size)
        
        # Use only the last time step output for classification
        x = x[:, -1, :]  # Take the last time step
        x = self.fc(x)
        return x