import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self, input_dim=550, num_classes=5):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.01),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def class_loss(classification, labels):
    classification_loss = F.cross_entropy(classification, labels, reduction='mean')
    return classification_loss

def weights_init(m, s):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            torch.manual_seed(s)
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

def preprocess_data(df):
    features = df.iloc[:, :-3].values
    labels = df.iloc[:, -3].map({'control': 0, 'mixture': 1, 'mix':1, 'sulfamethoxazole': 2, 'sulfapyridine': 3, 'sulfathiazole': 4}).values
    wavenumbers = df.columns[:-3].astype(float)
    concentration = df.iloc[:, -2].values
    return features, labels, wavenumbers, concentration

