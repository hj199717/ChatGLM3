# -*-coding:utf-8 -*-
'''
File       : vit.py
Time       : 2024/4/2 15:34
Author     : He Jia
version    : 1.0.0
Description: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import rearrange

# 定义 Vision Transformer 模型
class ViT(pl.LightningModule):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.5):
        super(ViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout), num_layers=depth)
        self.classification_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        embeddings = self.patch_to_embedding(patches)
        embeddings = rearrange(embeddings, 'b (h w) d -> b (h w) d')
        b, n, _ = embeddings.shape
        embeddings += self.pos_embedding[:, :(n + 1)]
        output = self.transformer(embeddings)
        output = output[:, 0]  # 取CLS token的输出作为分类结果
        output = self.classification_head(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 训练模型
model = ViT(image_size=224, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, mlp_dim=3072)
trainer = pl.Trainer(max_epochs=10, devices=2, accelerator="gpu")
trainer.fit(model, train_loader, val_loader)
