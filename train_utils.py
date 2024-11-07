import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import torch.optim as optim
import numpy as np
from tqdm import tqdm

PRINT_FREQU = 10

def compute_covariance(features): 
    ######################
    # Implement Coral loss
    ######################
    batch_size = features.size(0)
    features = features - torch.mean(features, dim=0)
    covariance = (1.0 / (batch_size - 1)) * torch.matmul(features.t(), features)
    return covariance


def train_baseline(model, source_loader, target_loader, args, device):
    """Standard source training"""
    print("\nTraining Baseline Model...")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Lists to store metrics
    train_losses = []
    source_accs = []
    target_accs = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for data, target in tqdm(source_loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Calculate and store metrics for each epoch
        avg_loss = total_loss / len(source_loader)
        train_losses.append(avg_loss)
        
        _, source_acc = evaluate(model, source_loader, device)
        _, target_acc = evaluate(model, target_loader, device)
        source_accs.append(source_acc)
        target_accs.append(target_acc)
        
        if (epoch + 1) % PRINT_FREQU == 0:
            print(f'\nEpoch: {epoch + 1}/{args.epochs}')
            print(f'Training Loss: {avg_loss:.4f} | Accuracies: Source = {source_acc:.4f}, Target = {target_acc:.4f}')
    
    # Save final model
    import os
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/baseline_final.pth')
    
    # Return dictionary with metrics
    return {
        'final_target_acc': target_accs[-1],
        'train_losses': train_losses,
        'source_accs': source_accs,
        'target_accs': target_accs
    }

def train_coral(model, source_loader, target_loader, args, device):
    """CORAL training"""
    print("\nTraining CORAL Model...")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Lists to store metrics
    train_losses = []
    source_accs = []
    target_accs = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for (source_data, source_target), (target_data, _) in zip(source_loader, target_loader):

            source_data = source_data.to(device)
            source_target = source_target.to(device)
            target_data = target_data.to(device)
            
            optimizer.zero_grad()
            
            # Extract features
            source_features = model.feature_extractor(source_data)
            target_features = model.feature_extractor(target_data)
            
            # Classification loss
            source_outputs = model.classifier(source_features)
            cls_loss = F.nll_loss(source_outputs, source_target)
            
            # CORAL loss
            source_cov = compute_covariance(source_features)
            target_cov = compute_covariance(target_features)
            coral_loss = torch.norm(source_cov - target_cov, p='fro')
            
            # Total loss
            loss = cls_loss + args.coral_weight * coral_loss 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate training and testing metrics
        avg_loss = total_loss / len(source_loader)
        train_losses.append(avg_loss)
        
        _, source_acc = evaluate(model, source_loader, device)
        _, target_acc = evaluate(model, target_loader, device)
        source_accs.append(source_acc)
        target_accs.append(target_acc)
        
        if (epoch + 1) % PRINT_FREQU == 0:
            print(f'\nEpoch: {epoch + 1}/{args.epochs}')
            print(f'Training Loss: {avg_loss:.4f} | Accuracies: Source = {source_acc:.4f}, Target = {target_acc:.4f}')
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/final_coral.pth')

    # Return Final target accuracy 
    return {
        'final_target_acc': target_accs[-1],
        'train_losses': train_losses,
        'source_accs': source_accs,
        'target_accs': target_accs
    }

def train_adversarial(model, source_loader, target_loader, args, device):
    """Adversarial training for domain adaptation"""
    print("\nTraining Adversarial Model...")
    
    # Domain discriminator network
    discriminator = nn.Sequential(
        nn.Linear(256, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 2),
    ).to(device)
    
    optimizer_g = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    # Lists to store metrics
    train_losses = []
    source_accs = []
    target_accs = []
    
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        total_loss_g = 0
        total_loss_d = 0
        
        for (source_data, source_target), (target_data, _) in zip(source_loader, target_loader):
            source_data = source_data.to(device)
            source_target = source_target.to(device)
            target_data = target_data.to(device)
            batch_size = source_data.size(0)
            
            # Train discriminator
            optimizer_d.zero_grad()
            
            source_features = model.feature_extractor(source_data).detach()
            target_features = model.feature_extractor(target_data).detach()
            
            source_domain = torch.zeros(batch_size).long().to(device)
            target_domain = torch.ones(batch_size).long().to(device)
            
            source_domain_pred = discriminator(source_features)
            target_domain_pred = discriminator(target_features)
            
            d_loss = F.cross_entropy(source_domain_pred, source_domain) + \
                    F.cross_entropy(target_domain_pred, target_domain)
            
            d_loss.backward()
            optimizer_d.step()
            total_loss_d += d_loss.item()
            
            # Train generator (feature extractor + classifier)
            optimizer_g.zero_grad()
            
            source_features = model.feature_extractor(source_data)
            target_features = model.feature_extractor(target_data)
            source_outputs = model.classifier(source_features)
            
            # Classification loss
            cls_loss = F.nll_loss(source_outputs, source_target)
            
            # Adversarial loss - try to fool discriminator
            source_domain_pred = discriminator(source_features)
            target_domain_pred = discriminator(target_features)
            
            # Fool discriminator by making it predict wrong domain
            g_loss = F.cross_entropy(source_domain_pred, target_domain) + \
                    F.cross_entropy(target_domain_pred, source_domain)
            
            loss_g = cls_loss + args.adversarial_weight * g_loss
            loss_g.backward()
            optimizer_g.step()
            total_loss_g += loss_g.item()
        
        # Calculate and store metrics
        avg_loss = (total_loss_g + total_loss_d) / len(source_loader)
        train_losses.append(avg_loss)
        
        _, source_acc = evaluate(model, source_loader, device)
        _, target_acc = evaluate(model, target_loader, device)
        source_accs.append(source_acc)
        target_accs.append(target_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch: {epoch + 1}/{args.epochs}')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Source Accuracy: {source_acc:.4f}')
            print(f'Target Accuracy: {target_acc:.4f}')
    
    # Save final model
    import os
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'discriminator': discriminator.state_dict()
    }, 'models/adversarial_final.pth')
    
    return {
        'final_target_acc': target_accs[-1],
        'train_losses': train_losses,
        'source_accs': source_accs,
        'target_accs': target_accs
    }

def train_adabn(model, source_loader, target_loader, args, device):
    """AdaBN with source training and target adaptation"""
    print("\nTraining AdaBN Model...")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Lists to store metrics
    train_losses = []
    source_accs = []
    target_accs = []
    
    # 1. Train on source
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for data, target in tqdm(source_loader, desc=f'Epoch {epoch + 1} (Source Training)'):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Calculate and store metrics
        avg_loss = total_loss / len(source_loader)
        train_losses.append(avg_loss)
        
        _, source_acc = evaluate(model, source_loader, device)
        _, target_acc = evaluate(model, target_loader, device)
        source_accs.append(source_acc)
        target_accs.append(target_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch: {epoch + 1}/{args.epochs}')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Source Accuracy: {source_acc:.4f}')
            print(f'Target Accuracy: {target_acc:.4f}')
    
    # 2. Adapt BN statistics on target domain
    model.train()  # Important: keep in train mode to update BN statistics
    print("\nAdapting BN statistics on target domain...")
    
    with torch.no_grad():  # No need to update weights, only BN statistics
        for _ in range(args.epochs):  # Usually fewer epochs needed for adaptation
            for data, _ in target_loader:
                data = data.to(device)
                # Forward pass updates BN stats
                model(data)
    
    # Final evaluation
    _, final_target_acc = evaluate(model, target_loader, device)
    print(f'\nFinal Target Accuracy after AdaBN: {final_target_acc:.4f}')
    
    # Save final model
    import os
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/adabn_final.pth')
    
    return {
        'final_target_acc': final_target_acc,
        'train_losses': train_losses,
        'source_accs': source_accs,
        'target_accs': target_accs
    }


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target).item()
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), correct / total