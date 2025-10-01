# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

def load_teacher(path, device):
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model



class Student20(nn.Module):
    def __init__(self):
        super(Student20, self).__init__()

        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(384)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(384, 832, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(832)
        self.pool3 = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(832, 100)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Student50(nn.Module):
    def __init__(self):
        super(Student50, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100)
        )
    def forward(self, x):
        return self.model(x)



class Generator(nn.Module):
    def __init__(self, noise_dim=128):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU()
        )

        self.gen = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.gen(x)


def imitation_loss(t_out, s_out, T=3.0):

    t_soft = F.softmax(t_out / T, dim=1)
    s_logsoft = F.log_softmax(s_out / T, dim=1)
    loss = F.kl_div(s_logsoft, t_soft, reduction='batchmean') * (T ** 2)
    return loss

def generation_loss(t_out, s_out, T=3.0):

    t_soft = F.softmax(t_out / T, dim=1)
    s_soft = F.softmax(s_out / T, dim=1)
    return -torch.mean(torch.sum(t_soft * torch.log((t_soft + 1e-6) / (s_soft + 1e-6)), dim=1))

def combined_loss(t_out, s_out, labels, T=3.0, alpha=0.7, beta=0.3):

    distill = imitation_loss(t_out, s_out, T)
    cls_loss = F.cross_entropy(s_out, labels)
    return alpha * distill + beta * cls_loss


def train_akd(teacher, student, generator, train_loader, device,
              epochs=100, k=3, batch_size=128, T=3.0,
              noise_dim=128, alpha=0.7, beta=0.3, lambda_fake=0.3):

    opt_s = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    opt_g = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    scheduler_s = CosineAnnealingLR(opt_s, T_max=epochs, eta_min=1e-6)
    scheduler_g = CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    checkpoint_dir = "saved_correct_code"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    for epoch in range(epochs):
        student.train()
        generator.train()
        total_fake_loss = 0.0
        total_gen_loss = 0.0
        batch_count = 0


        current_lambda = lambda_fake * min(1.0, epoch / (epochs * 0.3))

        for _, _ in train_loader:


            fake_loss = 0.0
            actual_batch_size = min(batch_size, 64)
            for _ in range(k):
                z = torch.randn(actual_batch_size, noise_dim).to(device)
                x_fake = generator(z)

                x_fake_up = F.interpolate(x_fake, size=(224,224), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    t_out_fake = teacher(x_fake_up)
                s_out_fake = student(x_fake_up)
                loss_fake = imitation_loss(t_out_fake, s_out_fake, T)
                opt_s.zero_grad()
                (current_lambda * loss_fake).backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_s.step()
                fake_loss += loss_fake.item()
            fake_loss /= k
            total_fake_loss += fake_loss


            if batch_count % 4 == 0:
                actual_batch_size = min(batch_size, 64)
                z = torch.randn(actual_batch_size, noise_dim).to(device)
                x_fake = generator(z)
                x_fake_up = F.interpolate(x_fake, size=(224,224), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    t_out_fake = teacher(x_fake_up)
                s_out_fake = student(x_fake_up)
                loss_g = generation_loss(t_out_fake, s_out_fake, T)
                opt_g.zero_grad()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                opt_g.step()
                total_gen_loss += loss_g.item()

            batch_count += 1

        scheduler_s.step()
        scheduler_g.step()

        avg_fake_loss = total_fake_loss / (batch_count / 2 + 1e-8)
        avg_gen_loss = total_gen_loss / (batch_count / 4 + 1e-8)

        print(f"Epoch {epoch+1}/{epochs}, Fake Loss: {avg_fake_loss:.4f}, Gen Loss: {avg_gen_loss:.4f}")


        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            student.eval()

            test_correct = 0

            print(f"Epoch {epoch+1} completed. (Evaluation on test set should be done outside the training loop.)")

        if (epoch + 1) % 10 == 0:
            student_ckpt = os.path.join(checkpoint_dir, f"{type(student).__name__}_epoch{epoch+1}.pth")
            generator_ckpt = os.path.join(checkpoint_dir, f"Generator_{type(student).__name__}_epoch{epoch+1}.pth")
            torch.save(student.state_dict(), student_ckpt)
            torch.save(generator.state_dict(), generator_ckpt)
            print(f"Checkpoint saved at epoch {epoch+1}:")
            print(f"  {student_ckpt}")
            print(f"  {generator_ckpt}")


def evaluate(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


    idx_20 = torch.randperm(len(test_set))[:int(0.2 * len(test_set))]
    idx_10 = idx_20[:int(0.5 * len(idx_20))]
    test_loader_20 = DataLoader(Subset(test_set, idx_20), batch_size=batch_size, shuffle=False)
    test_loader_10 = DataLoader(Subset(test_set, idx_10), batch_size=batch_size, shuffle=False)


    print("Loading teacher model...")
    try:
        teacher = load_teacher("best_resnet34_cifar100.pth", device)
        print("Teacher model loaded successfully!")
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        print("Using randomly initialized ResNet34 as teacher for demonstration.")
        teacher = models.resnet34(pretrained=False)
        teacher.fc = nn.Linear(teacher.fc.in_features, 100)
        teacher.to(device)
        teacher.eval()


    student_20 = Student20().to(device)
    student_50 = Student50().to(device)
    generator = Generator().to(device)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    teacher_params   = count_parameters(teacher)
    student20_params = count_parameters(student_20)
    student50_params = count_parameters(student_50)
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student 20 parameters: {student20_params:,} ({student20_params/teacher_params:.2%} of teacher)")
    print(f"Student 50 parameters: {student50_params:,} ({student50_params/teacher_params:.2%} of teacher)")

    epochs = 100


    print("\nTraining Student 20 Model (Data-free)...")
    train_akd(teacher, student_20, generator, train_loader, device,
              epochs=epochs, k=3, batch_size=batch_size, T=3.0,
              noise_dim=128, alpha=0.7, beta=0.3, lambda_fake=0.3)
    try:
        student_20.load_state_dict(torch.load(f"best_student_model_{type(student_20).__name__}.pth"))
    except:
        print("Warning: Could not load best model for Student20; using final state.")
    acc20_20, cm20_20 = evaluate(student_20, test_loader_20, device)
    acc20_10, cm20_10 = evaluate(student_20, test_loader_10, device)

    print("\nTraining Student 50 Model (Data-free)...")
    generator = Generator().to(device)
    train_akd(teacher, student_50, generator, train_loader, device,
              epochs=epochs, k=3, batch_size=batch_size, T=3.0,
              noise_dim=128, alpha=0.7, beta=0.3, lambda_fake=0.3)
    try:
        student_50.load_state_dict(torch.load(f"best_student_model_{type(student_50).__name__}.pth"))
    except:
        print("Warning: Could not load best model for Student50; using final state.")
    acc50_20, cm50_20 = evaluate(student_50, test_loader_20, device)
    acc50_10, cm50_10 = evaluate(student_50, test_loader_10, device)

    print("\n--- Evaluation Results ---")
    print("Student 20 Acc (20% split):", acc20_20*100)
    print("Student 20 Acc (10% split):", acc20_10*100)
    print("Student 50 Acc (20% split):", acc50_20*100)
    print("Student 50 Acc (10% split):", acc50_10*100)


    models_list = ['Student 20', 'Student 50']
    acc_20_split = [acc20_20, acc50_20]
    acc_10_split = [acc20_10, acc50_10]
    x = np.arange(len(models_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(x - width/2, acc_20_split, width, label='20% Test Split')
    rects2 = ax.bar(x + width/2, acc_10_split, width, label='10% Test Split')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison on Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list)
    ax.legend()
    ax.grid(True, alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.show()
