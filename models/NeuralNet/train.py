import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data.load import load_data
from model import RegressionDataset, MultipleRegression
import os

# Load data
df = load_data(num_rows=100000)

test = load_data(test=True, num_rows=100000)


# RM redundant features and fill NA
print('filling na')
y = df['booking_bool']
X = df.drop(['booking_bool','click_bool', 'position', 'gross_bookings_usd', 'date_time'], axis=1)
X = X.fillna(X.mean())

X_train, y_train = X, y
X_test = test.drop('date_time', axis=1)
X_test = X_test.fillna(X.mean()) #Mean of x or mean of x_test?

print(X.shape)

# Split for val data
y_test = np.zeros(X_test.shape[0])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=1)

# Normalize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

# Convert to float
y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

# Set the params
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.008
NUM_FEATURES = X.shape[1]

# Save destination
prefix = os.path.dirname(os.path.abspath(__file__))
best_model_path = prefix+"/saves/best_new"+".pt"

# Initialize datasets as dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

# I don't have CUDA but maybe you have
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize that model!
model = MultipleRegression(NUM_FEATURES)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)

# Keep track of training progress in dict
loss_stats = {
    'train': [],
    "val": []
}

# Save last validation loss to save best model
last_val_loss = 9999999999

# Let the training beginn
print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):
    
    # TRAINING
    train_epoch_loss = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
            
            val_epoch_loss += val_loss.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))                              
    
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')

        # Save the model if val loss is better
        if val_epoch_loss/len(val_loader)< last_val_loss:
            last_val_loss = val_epoch_loss/len(val_loader)
            torch.save(model.state_dict(), best_model_path)
        


# Visualize loss
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
plt.figure(figsize=(15,8))
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
plt.show()

# Test model
print("starting testing model")
y_pred_list = []

# Reload the model that was the best
model.load_state_dict(torch.load(best_model_path))

with torch.no_grad():
    model.eval()
    for X_batch, _ in val_loader: # Originally test-loader
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

print("Predicted and actual")
for y_pred, y_actual in zip(y_pred_list, y_test):
    print(y_pred, y_actual)
mse = mean_squared_error(y_test, y_pred_list)
mae = mean_absolute_error(y_test, y_pred_list)
r_square = r2_score(y_test, y_pred_list)
print("Mean Squared Error :",mse)
print("Mean Absolute Error :",mae)
print("R^2 :",r_square)

print("Values for avg as baseline would be")
y_pred_list = np.full(y_test.size, y_train.mean())
mse = mean_squared_error(y_test, y_pred_list)
mae = mean_absolute_error(y_test, y_pred_list)
r_square = r2_score(y_test, y_pred_list)
print("Mean Squared Error :",mse)
print("Mean Absolute Error :",mae)
print("R^2 :",r_square)
