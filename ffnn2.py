import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Define the network architecture
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# Prepare the data

train_data = pd.read_csv("train.csv", )
# if train_data["Embarked"] == "S":
#     train_data["Embarked"] = 1
# else:
#     train_data["Embarked"]=0
train_data.loc[train_data['Embarked'] == "S", "Embarked"] = 1
train_data.loc[train_data['Embarked'] == "C", "Embarked"] = 0
train_data.loc[train_data['Embarked'] == "Q", "Embarked"] = -1
train_data.loc[train_data['Sex'] == "male", "Sex"] = 1
train_data.loc[train_data['Sex'] == "female", "Sex"] = -1
# train_data.loc["Cabin"] = pd.ord(train_data['Cabin'])-101
train_data.drop('Name', axis=1, inplace=True)
train_data.drop('Ticket', axis=1, inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
# train_data.drop(train_data.columns[[0]], axis=0, inplace=True)

train_data["inp"]  = train_data.apply(lambda row: [row["PassengerId"], row["Pclass"], row["Sex"], row["Age"], row["SibSp"], row["Parch"], row["Embarked"]], axis=1)

train_data["out"] = train_data.apply(lambda row: [row["Survived"]], axis=1)

train_data.drop('Survived', axis=1, inplace=True)
train_data.drop('PassengerId', axis=1, inplace=True)
train_data.drop('Pclass', axis=1, inplace=True)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Sex', axis=1, inplace=True)
train_data.drop('Age', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('Fare', axis=1, inplace=True)
train_data.drop('Embarked', axis=1, inplace=True)

print(train_data)
train_data = train_data.values.tolist()

test_data = pd.read_csv("./test.csv")


input_size = 9
output_size = 1

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)



# x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
# y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Initialize the network
hidden_size = 4
net = FeedForwardNet(input_size, hidden_size, output_size)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Testing the trained network
with torch.no_grad():
    test_input = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predictions = net(test_input)
    predicted_classes = predictions.round()
    print("Predicted Classes:")
    print(predicted_classes)
