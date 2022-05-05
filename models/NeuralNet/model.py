import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_4 = nn.Linear(64, 128)
        self.layer_5 = nn.Linear(128, 16)
        self.drop_1 = nn.Dropout(p=0.2)
        self.layer_out = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.drop_1(x)
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.drop_1(x)
        x = self.relu(self.layer_out(x))
        return (x)

    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.drop_1(x)
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.drop_1(x)
        x = self.relu(self.layer_out(x))
        return (x)

# Create datasets
class RegressionDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def get_best_model(num_features):
    '''Get the best pre-trained model'''
    prefix = os.path.dirname(os.path.abspath(__file__))
    best_model_path = prefix+"/saves/best_new"+".pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultipleRegression(num_features)
    model.to(device)
    model.load_state_dict(torch.load(best_model_path))
    return model