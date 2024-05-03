import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, GRU, Linear, BCELoss, Conv2d, ZeroPad2d
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import torch.utils.data as Data
import wandb

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.embedder1 = Embedding(num_embeddings=316, embedding_dim=16)
        self.embedder2 = Embedding(num_embeddings=289, embedding_dim=8)

        self.cnn1_1 = Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 16), stride=1, padding=(1, 0))
        self.cnn1_2 = Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=1)
        self.cnn1_3 = Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 16), stride=1, padding=(2, 0))

        self.cnn2 = Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 8), stride=4)

        self.gru = GRU(input_size=512, hidden_size=100, bidirectional=True, batch_first=True)

        self.lin1 = Linear(200, 64)
        self.lin2 = Linear(64, 32)
        self.lin3 = Linear(32, 1)

    def forward(self, data):
        x_name, x_behavior = data.split([1000, 4000], 1) 

        x_name = self.embedder1(x_name)
        x_behavior = self.embedder2(x_behavior)

        x_name = x_name.unsqueeze(1)
        x_behavior = x_behavior.unsqueeze(1)

        pad = ZeroPad2d(padding=(0, 0, 2, 1))
        x_name_pad = pad(x_name)

        x_name_cnn1 = F.relu(self.cnn1_1(x_name)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn2 = F.relu(self.cnn1_2(x_name_pad)).squeeze(-1).permute(0, 2, 1)
        x_name_cnn3 = F.relu(self.cnn1_3(x_name)).squeeze(-1).permute(0, 2, 1)

        x_behavior = F.relu(self.cnn2(x_behavior)).squeeze(-1).permute(0, 2, 1)

        x = torch.cat([x_name_cnn1, x_name_cnn2, x_name_cnn3, x_behavior], dim=-1)

        x, h_n = self.gru(x)
	
        output_fw = h_n[-2, :, :]
        output_bw = h_n[-1, :, :]

        x = torch.cat([output_fw, output_bw], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.sigmoid(self.lin3(x))

        return x

def train(epoch, loader):
    model.train()

    loss_all = 0
    for step, (b_x, b_y) in enumerate(loader):
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        optimizer.zero_grad()
        output = model(b_x)
        loss = F.binary_cross_entropy(output, b_y)
        loss.backward()
        loss_all += b_x.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_x)

def test(loader):
    model.eval()
    label_list = []
    pred_list = []
    for step, (b_x, b_y) in enumerate(loader):
        b_x = b_x.to(device)
        pred = model(b_x)
        pred = torch.where(pred >= 0.5, torch.ones_like(pred), torch.zeros_like(pred))

        label_list_batch = b_y.to('cpu').detach().numpy().tolist()
        pred_list_batch = pred.to('cpu').detach().numpy().tolist()
        for label_item in label_list_batch:
            label_list.append(label_item)
        for pred_item in pred_list_batch:
            pred_list.append(pred_item)

    y_true = np.asarray(label_list)
    y_pred = np.asarray(pred_list)
    _val_confusion_matrix = confusion_matrix(y_true, y_pred)
    _val_acc = accuracy_score(y_true, y_pred)
    _val_precision = precision_score(y_true, y_pred)
    _val_recall = recall_score(y_true, y_pred)
    _val_f1 = f1_score(y_true, y_pred)

    wandb.log({"Test Accuracy": _val_acc, "Test Precision": _val_precision, "Test Recall": _val_recall, "Test F1": _val_f1})
    return _val_confusion_matrix, _val_acc, _val_precision, _val_recall, _val_f1

wandb.init(project="model_GRU", config={
    "learning_rate": 0.0001,
    "architecture": "Net",
    "dataset": "Custom",
    "epochs": 50,
})

train_data = np.load('../dataset/report_dataset_train.npz', allow_pickle=True)
train_x_name = train_data['x_name']
train_x_semantic = train_data['x_semantic']
train_y = train_data['y']

test_data = np.load('../dataset/report_dataset_test.npz', allow_pickle=True)
test_x_name = test_data['x_name']
test_x_semantic = test_data['x_semantic']
test_y = test_data['y']
train_x = np.concatenate([train_x_name, train_x_semantic], 1)
test_x = np.concatenate([test_x_name, test_x_semantic], 1)

train_xt = torch.from_numpy(train_x)
test_xt = torch.from_numpy(test_x)
train_yt = torch.from_numpy(train_y.astype(np.float32))
test_yt = torch.from_numpy(test_y.astype(np.float32))

train_data = Data.TensorDataset(train_xt, train_yt)
test_data = Data.TensorDataset(test_xt, test_yt)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=1,
)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=64,
    num_workers=1,
)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

for epoch in range(1, wandb.config.epochs + 1):
    loss = train(epoch, train_loader)
    con, acc, precision, recall, f1 = test(train_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {acc:.5f}, Train Precision: {precision:.5f}, Train Recall: {recall:.5f}, Train F1: {f1:.5f}')
    wandb.log({"Epoch Loss": loss, "Epoch Accuracy": acc, "Epoch Precision": precision, "Epoch Recall": recall, "Epoch F1": f1})

con, acc, precision, recall, f1 = test(test_loader)
print("==================================")
print('Test result:')
print('Accuracy:', acc)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print(con)

torch.save(model, './model_wandb.pkl')
wandb.finish()