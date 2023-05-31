import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    def __init__(self, dataframe):
        self.inputs = dataframe["inp"].tolist()
        self.targets = dataframe["out"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index], dtype=torch.float32)
        targets = torch.tensor(self.targets[index], dtype=torch.float32)
        return inputs, targets


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

m_val = train_data["PassengerId"].mean()
r_val = train_data["PassengerId"].max()-train_data["PassengerId"].min()
train_data["PassengerId"] = (train_data["PassengerId"]-m_val)/r_val

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

train_data["inp"]  = train_data.apply(lambda row: [row["PassengerId"], row["Pclass"], row["Sex"], row["Age"], row["SibSp"], row["Parch"], row["Fare"], row["Embarked"]], axis=1)

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

# print(train_data)
# train_data = train_data.values.tolist()

train_dataset = TitanicDataset(train_data)

test_data = pd.read_csv("./test.csv")


input_size = 8
output_size = 1
batch_size = 50
n_epoch = 2

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)




class FFNNmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNNmodel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)

        return out
    

input_dim = 8
hidden_dim = 10
output_dim = 1


model = FFNNmodel(input_dim, hidden_dim, output_dim)

criterion = nn.BCEWithLogitsLoss()


learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# print(list(model.parameters())[2].size())
# for epoch in range(n_epoch):
# it = iter(train_loader)
# print(it.__next__())
# print(iter(train_loader).__next__())

iter = 0
# print(iter(train_loader).__next__())

for epoch in range(n_epoch):
    for (inp, out) in train_loader:
        # l = len(PassId)
        # PassId = PassId.view(-1,1)
        # sur = sur.view(-1,1)
        # Pclass = Pclass.view(-1,1)
        # Sex = Sex.view(-1,1)
        # Age = Age.view(-1,1)
        # SibSp = SibSp.view(-1,1)
        # Rarch = Rarch.view(-1,1)
        # Fare = Fare.view(-1,1)
        # Emb = Emb.view(-1,1)
        # t = torch.cat((PassId, Pclass, Sex, Age, SibSp, Rarch, Fare, Emb), dim=1)
        # # print(t)
        # print(inp)
        # optimzer.zero_grad()

        outputs = model(inp)
        print(outputs)
        loss = criterion(outputs, out)

        optimizer.zero_grad()        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        iter += 1
        
        print(loss)
