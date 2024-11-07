import matplotlib.pyplot as plt


def plot_training_curves(results, save=False, show=True):
    """
    Plot training curves from training results dictionary for all models.
    
    Args:
        results (dict): Dictionary containing models and their metrics
            results['model_name'] contains:
                - 'train_losses': list of training losses
                - 'source_accs': list of source accuracies
                - 'target_accs': list of target accuracies
        save (bool): If True, saves plots to plots/ directory
        show (bool): If True, displays plots using plt.show()
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    
    if save:
        os.makedirs('plots', exist_ok=True)
        
    for model_name, model_results in results.items():
        if all(k in model_results for k in ['train_losses', 'source_accs', 'target_accs']):
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot training loss
            ax1.plot(model_results['train_losses'], label='Training Loss')
            ax1.set_title(f'{model_name} - Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracies on same plot
            ax2.plot(model_results['source_accs'], label='Source Accuracy')
            ax2.plot(model_results['target_accs'], label='Target Accuracy')
            ax2.set_title(f'{model_name} - Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(f'plots/{model_name}_plot.png')
            
            if show:
                plt.show()
            else:
                plt.close()


def set_seed(seed):
    # Python's random module
    import random
    random.seed(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set deterministic backend for CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set worker seeds
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    # Create generator for DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return seed_worker, generator