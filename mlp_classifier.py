import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, log_loss
from torch.utils.data import DataLoader
import os
import random
from datetime import datetime
import logging
import wandb
import joblib

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def calculate_metrics(y_true, y_pred, y_proba):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    loss = log_loss(y_true, y_proba)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        specificity = 0.0 if np.sum(y_true == 0) == 0 else 1.0

    return float(accuracy), float(sensitivity), float(specificity), float(f1), float(loss)

def dataloader_to_numpy(loader: DataLoader):
    X_list, y_list = [], []
    for inputs, targets in loader:
        X_list.append(inputs.numpy())
        y_list.append(targets.numpy())
    
    if not X_list:
        return np.array([]), np.array([])
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def classifier(train_loader, val_loader, test_loader, config, save_path=None):
    # data
    X_train, y_train = dataloader_to_numpy(train_loader)
    X_val, y_val = dataloader_to_numpy(val_loader)
    X_test, y_test = dataloader_to_numpy(test_loader)

    # data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_test

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), 
        activation='tanh',
        solver='sgd',
        alpha=0.001,                # L2 regularization 
        batch_size='auto',
        learning_rate='adaptive',   
        learning_rate_init=0.02,
        max_iter=200,            
        shuffle=True,
        random_state=config.seed,
        early_stopping=False,    
        tol=1e-4
    )
    
    # train
    print("Training MLP model...")
    model.fit(X_train_scaled, y_train)
    print(f"Training complete. Iterations: {model.n_iter_}, Loss: {model.loss_:.4f}")

    # evaluate
    results = {}
    datasets = {
        'train': (X_train_scaled, y_train), 
        'val': (X_val_scaled, y_val), 
        'test': (X_test_scaled, y_test)
    }

    for name, (X, y) in datasets.items():
        if len(X) > 0:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            
            acc, sens, spec, f1, loss = calculate_metrics(y, y_pred, y_proba)
            
            results[f'{name}_loss'] = loss
            results[f'{name}_acc'] = acc
            results[f'{name}_sens'] = sens
            results[f'{name}_spec'] = spec
            results[f'{name}_f1'] = f1
        else:
            results.update({f'{name}_{m}': 0.0 for m in ['loss', 'acc', 'sens', 'spec', 'f1']})

    # save model
    if save_path:
        joblib.dump(model, save_path)
        print(f"Classifier model saved to: {save_path}")

    return results

def main():
    from ppo_config import get_ppo_config
    from load_brain_data import BrainDataset

    args = get_ppo_config()
    seed_everything(args.seed)

    # Dataloaders
    train_loaders = {
        f: DataLoader(BrainDataset(txt, args.data_dir), batch_size=len(BrainDataset(txt, args.data_dir)))
        for f, txt in args.train_folds.items()
    }
    val_loaders = {
        f: DataLoader(BrainDataset(txt, args.data_dir), batch_size=len(BrainDataset(txt, args.data_dir)))
        for f, txt in args.valid_folds.items()
    }
    test_loaders = {
        f: DataLoader(BrainDataset(txt, args.data_dir), batch_size=len(BrainDataset(txt, args.data_dir)))
        for f, txt in args.test_folds.items()
    }

    # logging
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'classifier_metrics_{now}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info("CONFIG ▶ %s", vars(args))
    print("CONFIG ▶", vars(args))

    all_fold_metrics = []
    for fold in range(1, 6):
        print(f"\n===== Starting Fold {fold}/5 =====")
        
        train_loader = train_loaders[fold]
        val_loader = val_loaders[fold]
        test_loader = test_loaders[fold]

        save_dir = "classifier_baseline_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"classifier_baseline_fold{fold}.joblib")

        fold_metrics = classifier(
            train_loader, val_loader, test_loader,
            config=args,
            save_path=save_path,
        )
        fold_metrics['fold'] = fold
        all_fold_metrics.append(fold_metrics)
        
        print(f"Fold {fold} Results: {fold_metrics}")
        logging.info(f"Fold {fold} Results: {fold_metrics}")

    avg_metrics, std_metrics = {}, {}
    metric_keys = [k for k in all_fold_metrics[-1].keys() if k != 'fold']
    
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        avg_metrics[f"avg_{key}"] = float(np.mean(values))
        std_metrics[f"std_{key}"] = float(np.std(values))

    print("\n===== 5-Fold CV Results =====")
    print("Average Metrics:", avg_metrics)
    print("Std Dev Metrics:", std_metrics)
    logging.info("Average metrics across folds: %s", avg_metrics)
    logging.info("Standard deviation across folds: %s", std_metrics)

if __name__ == "__main__":
    main()
