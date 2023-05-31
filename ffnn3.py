import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class TitanicDataset(Dataset):
    def __init__(self, dataframe):
        self.inputs = dataframe.drop("Survived", axis=1).values.astype(float)
        self.targets = dataframe["Survived"].values.astype(float)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index], dtype=torch.float32)
        targets = torch.tensor(self.targets[index], dtype=torch.float32)
        return inputs, targets


# Load and preprocess the dataset
train_data = pd.read_csv("train.csv")
train_data.loc[train_data['Embarked'] == "S", "Embarked"] = 1
train_data.loc[train_data['Embarked'] == "C", "Embarked"] = 0
train_data.loc[train_data['Embarked'] == "Q", "Embarked"] = -1
train_data.loc[train_data['Sex'] == "male", "Sex"] = 1
train_data.loc[train_data['Sex'] == "female", "Sex"] = -1
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].mean(), inplace=True)

# m_val = train_data["PassengerId"].mean()
# r_val = train_data["PassengerId"].max()-train_data["PassengerId"].min()
# train_data["PassengerId"] = (train_data["PassengerId"]-m_val)/r_val

m_val = train_data["Pclass"].mean()
r_val = train_data["Pclass"].max()-train_data["Pclass"].min()
train_data["Pclass"] = (train_data["Pclass"]-m_val)/r_val

m_val = train_data["SibSp"].mean()
r_val = train_data["SibSp"].max()-train_data["SibSp"].min()
train_data["SibSp"] = (train_data["SibSp"]-m_val)/r_val
m_val = train_data["Sex"].mean()
r_val = train_data["Sex"].max()-train_data["Sex"].min()
train_data["Sex"] = (train_data["Sex"]-m_val)/r_val

m_val = train_data["Age"].mean()
train_data.loc[train_data['Age'] == "", "Age"] = m_val
r_val = train_data["Age"].max()-train_data["Age"].min()
train_data["Age"] = (train_data["Age"]-m_val)/r_val

m_val = train_data["Parch"].mean()
r_val = train_data["Parch"].max()-train_data["Parch"].min()
train_data["Parch"] = (train_data["Parch"]-m_val)/r_val

m_val = train_data["Fare"].mean()
r_val = train_data["Fare"].max()-train_data["Fare"].min()
train_data["Fare"] = (train_data["Fare"]-m_val)/r_val

m_val = train_data["Embarked"].mean()
r_val = train_data["Embarked"].max()-train_data["Embarked"].min()
train_data["Embarked"] = (train_data["Embarked"]-m_val)/r_val

# print(train_data)
train_dataset = TitanicDataset(train_data)

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,hidden_dim2,  output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out

input_dim = 7  # Number of input features
hidden_dim = 20  # Number of neurons in the hidden layer
hidden_dim2 = 20
output_dim = 1  # Single output for survival prediction

model = MLPModel(input_dim, hidden_dim , hidden_dim2, output_dim)

# Define the training parameters
learning_rate = 0.0005
batch_size = 32
num_epochs = 100

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # print(inputs)
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model
train_data = pd.read_csv("test.csv")
# test_data.loc[test_data['Embarked'] == "S", "Embarked"] = 1
# test_data.loc[test_data['Embarked'] == "C", "Embarked"] = 0
# test_data.loc[test_data['Embarked'] == "Q", "Embarked"] = -1
# test_data.loc[test_data['Sex'] == "male", "Sex"] = 1
# test_data.loc[test_data['Sex'] == "female", "Sex"] = -1
# test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1,

# train_data = pd.read_csv("train.csv").dropna()
train_data.loc[train_data['Embarked'] == "S", "Embarked"] = 1
train_data.loc[train_data['Embarked'] == "C", "Embarked"] = 0
train_data.loc[train_data['Embarked'] == "Q", "Embarked"] = -1
train_data.loc[train_data['Sex'] == "male", "Sex"] = 1
train_data.loc[train_data['Sex'] == "female", "Sex"] = -1
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].mean(), inplace=True)

# m_val = train_data["PassengerId"].mean()
# r_val = train_data["PassengerId"].max()-train_data["PassengerId"].min()
# train_data["PassengerId"] = (train_data["PassengerId"]-m_val)/r_val

m_val = train_data["Pclass"].mean()
r_val = train_data["Pclass"].max()-train_data["Pclass"].min()
train_data["Pclass"] = (train_data["Pclass"]-m_val)/r_val

m_val = train_data["SibSp"].mean()
r_val = train_data["SibSp"].max()-train_data["SibSp"].min()
train_data["SibSp"] = (train_data["SibSp"]-m_val)/r_val
m_val = train_data["Sex"].mean()
r_val = train_data["Sex"].max()-train_data["Sex"].min()
train_data["Sex"] = (train_data["Sex"]-m_val)/r_val

m_val = train_data["Age"].mean()
# train_data.loc[train_data['Age'] == None, "Age"] = m_val
r_val = train_data["Age"].max()-train_data["Age"].min()
train_data["Age"] = (train_data["Age"]-m_val)/r_val

m_val = train_data["Parch"].mean()
r_val = train_data["Parch"].max()-train_data["Parch"].min()
train_data["Parch"] = (train_data["Parch"]-m_val)/r_val

m_val = train_data["Fare"].mean()
r_val = train_data["Fare"].max()-train_data["Fare"].min()
train_data["Fare"] = (train_data["Fare"]-m_val)/r_val

m_val = train_data["Embarked"].mean()
r_val = train_data["Embarked"].max()-train_data["Embarked"].min()
train_data["Embarked"] = (train_data["Embarked"]-m_val)/r_val

print(train_data)

train_dataset = TitanicDataset(train_data)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# for inputs, vals in train_loader:
#     outputs = model(inputs)
#     pred = torch.sigmoid(outputs.squeeze())
#     print(inputs)

model.eval()
predictions = []
with torch.no_grad():
    for inputs, i in test_loader:
        # print(inputs)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs.squeeze())
        predictions.extend(preds.tolist())

# Print the predicted survival probabilities
train_data=train_data.values.tolist()
print(train_data)
cor = 0
tot = 0
for i, pred in enumerate(predictions):
        print(f"Passenger {i+1}: {round(pred)} / {train_data[i][0]}")
        if(round(pred) == train_data[i][0]):
            cor += 1
        tot+=1
    
print(cor/tot)
    