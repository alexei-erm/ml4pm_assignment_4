import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def get_predictions(model, loader, device):
    """Get all predictions and true labels from a data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(cm, class_names, title):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def analyze_per_class_performance(models_dict, source_loader, target_loader, device, 
                                class_names=None, save_plots=True):
    """
    Analyze per-class performance for multiple models.
    
    Args:
        models_dict: Dictionary of trained models {'model_name': model}
        source_loader: DataLoader for source domain
        target_loader: DataLoader for target domain
        device: torch device
        class_names: List of class names (optional)
        save_plots: Whether to save plots to disk
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\nAnalyzing {model_name}...")
        model_results = {}
        
        # Source domain analysis
        source_preds, source_labels = get_predictions(model, source_loader, device)
        source_cm = confusion_matrix(source_labels, source_preds)
        source_report = classification_report(source_labels, source_preds, 
                                           target_names=class_names, 
                                           output_dict=True)
        
        # Target domain analysis
        target_preds, target_labels = get_predictions(model, target_loader, device)
        target_cm = confusion_matrix(target_labels, target_preds)
        target_report = classification_report(target_labels, target_preds, 
                                           target_names=class_names, 
                                           output_dict=True)
        
        # Plot confusion matrices
        plt.figure(figsize=(20, 8))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(source_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'{model_name} - Source Domain Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(target_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'{model_name} - Target Domain Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_plots:
            plt.savefig(f'plots/{model_name}_confusion_matrices.png')
        plt.close()
        
        # Create performance comparison DataFrame
        source_metrics = pd.DataFrame(source_report).transpose()
        target_metrics = pd.DataFrame(target_report).transpose()
        
        metrics_comparison = pd.DataFrame({
            'Source_Precision': source_metrics['precision'],
            'Source_Recall': source_metrics['recall'],
            'Source_F1': source_metrics['f1-score'],
            'Target_Precision': target_metrics['precision'],
            'Target_Recall': target_metrics['recall'],
            'Target_F1': target_metrics['f1-score']
        })
        
        # Plot per-class performance comparison
        metrics_to_plot = ['F1', 'Precision', 'Recall']
        plt.figure(figsize=(15, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1, 3, i+1)
            source_metric = metrics_comparison[f'Source_{metric}']
            target_metric = metrics_comparison[f'Target_{metric}']
            
            class_indices = metrics_comparison.index[:len(class_names)]
            
            plt.bar(np.arange(len(class_names)) - 0.2, 
                   source_metric[class_indices], 0.4, label='Source')
            plt.bar(np.arange(len(class_names)) + 0.2, 
                   target_metric[class_indices], 0.4, label='Target')
            
            plt.title(f'{metric} Score by Class')
            plt.xlabel('Class')
            plt.ylabel(f'{metric} Score')
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'plots/{model_name}_metrics_comparison.png')
        plt.close()
        
        # Store results
        model_results = {
            'source_confusion_matrix': source_cm,
            'target_confusion_matrix': target_cm,
            'source_report': source_report,
            'target_report': target_report,
            'metrics_comparison': metrics_comparison
        }
        
        results[model_name] = model_results
        
        # Print summary metrics
        print(f"\n{model_name} Summary:")
        print("\nSource Domain Performance:")
        print(f"Overall Accuracy: {source_report['accuracy']:.4f}")
        print("\nTarget Domain Performance:")
        print(f"Overall Accuracy: {target_report['accuracy']:.4f}")
        print("\nPer-class F1 scores (Source → Target):")
        for cls in class_names:
            source_f1 = source_report[cls]['f1-score']
            target_f1 = target_report[cls]['f1-score']
            print(f"{cls}: {source_f1:.4f} → {target_f1:.4f}")
    
    return results

def print_domain_shift_analysis(results, class_names=None):
    """Analyze which classes are most affected by domain shift."""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]
    
    for model_name, model_results in results.items():
        source_report = model_results['source_report']
        target_report = model_results['target_report']
        
        # Calculate performance drops
        performance_drops = []
        for cls in class_names:
            source_f1 = source_report[cls]['f1-score']
            target_f1 = target_report[cls]['f1-score']
            drop = source_f1 - target_f1
            performance_drops.append((cls, drop))
        
        # Sort by performance drop
        performance_drops.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{model_name} - Classes most affected by domain shift:")
        for cls, drop in performance_drops:
            print(f"{cls}: F1-score drop of {drop:.4f}")




import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(model, loader, device):
    """Extract features from the feature extractor part of the model."""
    model.eval()
    features = []
    labels = []
    domains = []  # 0 for source, 1 for target
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            # Get features before classifier
            feat = model.feature_extractor(data)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
            # Add domain labels (same length as batch)
            domains.extend([0] * len(target))
    
    return np.vstack(features), np.hstack(labels), np.array(domains)

def visualize_features(features, labels, domains, method='tsne', title='', save_path=None):
    """
    Visualize features using t-SNE or PCA.
    
    Args:
        features: numpy array of features
        labels: numpy array of class labels
        domains: numpy array of domain labels (0=source, 1=target)
        method: 'tsne' or 'pca'
        title: plot title
        save_path: path to save the plot
    """
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2)
    
    embedded = reducer.fit_transform(features)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Color by class
    scatter1 = ax1.scatter(embedded[:, 0], embedded[:, 1], 
                          c=labels, cmap='tab10',
                          alpha=0.6)
    ax1.set_title(f'{title}\nColored by Class')
    legend1 = ax1.legend(*scatter1.legend_elements(),
                        loc="center left", bbox_to_anchor=(1, 0.5),
                        title="Classes")
    ax1.add_artist(legend1)
    
    # Plot 2: Color by domain
    scatter2 = ax2.scatter(embedded[:, 0], embedded[:, 1],
                          c=domains, cmap='coolwarm',
                          alpha=0.6)
    ax2.set_title(f'{title}\nColored by Domain (Red=Source, Blue=Target)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def calculate_domain_statistics(features_source, features_target):
    """Calculate statistics about domain alignment."""
    # Calculate means
    source_mean = np.mean(features_source, axis=0)
    target_mean = np.mean(features_target, axis=0)
    mean_distance = np.linalg.norm(source_mean - target_mean)
    
    # Calculate covariances
    source_cov = np.cov(features_source.T)
    target_cov = np.cov(features_target.T)
    cov_frob_distance = np.linalg.norm(source_cov - target_cov, ord='fro')
    
    # Calculate MMD (Maximum Mean Discrepancy) - simplified version
    def mmd(x, y):
        xx = np.mean(np.dot(x, x.T))
        yy = np.mean(np.dot(y, y.T))
        xy = np.mean(np.dot(x, y.T))
        return xx + yy - 2*xy
    
    mmd_distance = mmd(features_source, features_target)
    
    return {
        'mean_distance': mean_distance,
        'covariance_distance': cov_frob_distance,
        'mmd': mmd_distance
    }

def analyze_feature_space(models_dict, source_loader, target_loader, device, 
                         save_dir='plots/feature_space/'):
    """Comprehensive feature space analysis for all models."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\nAnalyzing feature space for {model_name}...")
        
        # Extract features
        source_features, source_labels, source_domains = extract_features(
            model, source_loader, device)
        target_features, target_labels, target_domains = extract_features(
            model, target_loader, device)
        
        # Combine features for visualization
        combined_features = np.vstack([source_features, target_features])
        combined_labels = np.hstack([source_labels, target_labels])
        combined_domains = np.hstack([source_domains, np.ones_like(target_domains)])
        
        # Visualize using t-SNE
        visualize_features(combined_features, combined_labels, combined_domains,
                         method='tsne',
                         title=f'{model_name} Feature Space (t-SNE)',
                         save_path=os.path.join(save_dir, f'{model_name}_tsne.png'))
        
        # Visualize using PCA
        visualize_features(combined_features, combined_labels, combined_domains,
                         method='pca',
                         title=f'{model_name} Feature Space (PCA)',
                         save_path=os.path.join(save_dir, f'{model_name}_pca.png'))
        
        # Calculate domain statistics
        stats = calculate_domain_statistics(source_features, target_features)
        results[model_name] = stats
        
        print(f"\n{model_name} Domain Statistics:")
        print(f"Mean Distance: {stats['mean_distance']:.4f}")
        print(f"Covariance Distance: {stats['covariance_distance']:.4f}")
        print(f"MMD Distance: {stats['mmd']:.4f}")
    
    # Create comparative visualizations
    plt.figure(figsize=(12, 6))
    
    # Plot mean distances
    plt.subplot(131)
    mean_distances = [results[model]['mean_distance'] for model in models_dict.keys()]
    plt.bar(models_dict.keys(), mean_distances)
    plt.title('Mean Feature Distance')
    plt.xticks(rotation=45)
    
    # Plot covariance distances
    plt.subplot(132)
    cov_distances = [results[model]['covariance_distance'] for model in models_dict.keys()]
    plt.bar(models_dict.keys(), cov_distances)
    plt.title('Covariance Distance')
    plt.xticks(rotation=45)
    
    # Plot MMD
    plt.subplot(133)
    mmd_distances = [results[model]['mmd'] for model in models_dict.keys()]
    plt.bar(models_dict.keys(), mmd_distances)
    plt.title('MMD Distance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'domain_distances_comparison.png'))
    plt.close()
    
    return results

def plot_class_separation(features, labels, domains, save_path=None):
    """Analyze and plot class separation in feature space."""
    # Calculate average intra-class distance for each domain
    unique_classes = np.unique(labels)
    source_mask = domains == 0
    target_mask = domains == 1
    
    intra_class_distances = {
        'source': [],
        'target': []
    }
    
    for cls in unique_classes:
        # Source domain
        cls_mask_source = (labels == cls) & source_mask
        if np.sum(cls_mask_source) > 1:
            cls_features = features[cls_mask_source]
            distances = np.mean([np.mean(np.linalg.norm(f - cls_features, axis=1)) 
                               for f in cls_features])
            intra_class_distances['source'].append(distances)
        
        # Target domain
        cls_mask_target = (labels == cls) & target_mask
        if np.sum(cls_mask_target) > 1:
            cls_features = features[cls_mask_target]
            distances = np.mean([np.mean(np.linalg.norm(f - cls_features, axis=1)) 
                               for f in cls_features])
            intra_class_distances['target'].append(distances)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    x = np.arange(len(unique_classes))
    width = 0.35
    
    plt.bar(x - width/2, intra_class_distances['source'], width, label='Source')
    plt.bar(x + width/2, intra_class_distances['target'], width, label='Target')
    
    plt.xlabel('Class')
    plt.ylabel('Average Intra-class Distance')
    plt.title('Class Separation Analysis')
    plt.xticks(x, [f'Class {i}' for i in unique_classes])
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()