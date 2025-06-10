import json
import torch
from datetime import datetime
from src.data import train_val_split, make_eeg_dataloader_from_dict
from src.hyperparam_search.unsupervised_ae_search import unsupervised_ae_search
from src.hyperparam_search.semisupervised_ae_search import semisupervised_ae_search
from src.hyperparam_search.hyperparam_configs import unsup_ae_search_space, semi_ae_search_space
from src.data import load_train_test_data
from src.configs import  MODELS_DIR, HYPERPARAM_RESULTS_DIR
from src.utils import relative_path_str

def run_search():
    # Load data
    all_data = load_train_test_data()
    full_train_dict = all_data['train']
    
    # Split
    train_dict, val_dict = train_val_split(full_train_dict)
    
    # Make DataLoaders
    train_loader = make_eeg_dataloader_from_dict(train_dict)
    val_loader = make_eeg_dataloader_from_dict(val_dict, shuffle=False)
    
    # 1. Hyper param search : Unsupervised autoencoder
    best_u_model, best_u_config, best_u_score, u_results = unsupervised_ae_search(
        train_loader, val_loader, unsup_ae_search_space
    )
    save_search_results(u_results, "unsupervised_ae")
    save_best_model(best_u_model, best_u_config, best_u_score, "unsupervised_ae")
    
    # 2. Hyperparameter search: Semi-supervised autoencoder
    best_s_model, best_s_config, best_s_score, s_results = semisupervised_ae_search(
        train_loader, val_loader, semi_ae_search_space
    )
    save_search_results(s_results, "semisupervised_ae")
    save_best_model(best_s_model, best_s_config, best_s_score, "semisupervised_ae")

    # 3. Hyper param search : Semi-supervised RVAE
    # TO-DO: Implement semi-supervised RVAE search

    # 4. Hyper param search: Unsupervised RVAE
    # TO-DO

def save_search_results(results, model_type):
    """Save hyperparameter search results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_hyperparam_search_{timestamp}.json"
    filepath = HYPERPARAM_RESULTS_DIR / filename
    
    # Convert numpy types to native Python types for JSON serialization
    json_results = []
    for result in results:
        json_result = {
            'hyperparams': result['hyperparams'],
            'val_acc': float(result['val_acc']),
            'val_roc_auc': float(result.get('val_roc_auc', 0))  # Add ROC-AUC if available
        }
        json_results.append(json_result)
    
    with open(filepath, 'w') as f:
        json.dump({
            'model_type': model_type,
            'search_space': unsup_ae_search_space if 'unsupervised' in model_type else semi_ae_search_space,
            'timestamp': timestamp,
            'results': json_results
        }, f, indent=2)
    
    print(f"Hyper param search results saved to: {relative_path_str(filepath)}")
    return filepath

def save_best_model(model, config, score, model_type):
    """Save the best model and its configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model state dict
    model_filename = f"best_{model_type}_{timestamp}.pth"
    model_path = MODELS_DIR / model_filename
    torch.save(model.state_dict(), model_path)
    
    # Save config and metadata
    config_filename = f"best_{model_type}_{timestamp}_config.json"
    config_path = MODELS_DIR / config_filename
    
    config_data = {
        'model_type': model_type,
        'best_config': config,
        'best_val_score': float(score),
        'timestamp': timestamp,
        'model_file': model_filename,
        'architecture': {
            'input_dim': model.encoder[0].in_features if hasattr(model, 'encoder') else None,
            'latent_dim': config.get('latent_dim'),
            'model_class': model.__class__.__name__
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Best model saved to: {relative_path_str(model_path)}")
    print(f"Best config saved to: {relative_path_str(config_path)}")
    return model_path, config_path

if __name__ == "__main__":
    run_search()