import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, precision_score
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def train_and_evaluate_model_with_cv(data, labels):

    # Set device (GPU if available)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_labels=1
    batch_size=33
    num_epochs=15

    # Initialize tokenizer and model using AutoTokenizer and AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("../deberta-v3-base")

    # Initialize lists to track metrics
    accuracy_list, sensitivity_list, specificity_list, mcc_list, auc_list, f1_list, precision_list = [], [], [], [], [], [], []

    # Preprocess data
    encoded_texts = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_texts['input_ids'].to(device)
    attention_mask = encoded_texts['attention_mask'].to(device)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    # Define KFold cross-validation
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(input_ids)):
        print(f"##############\nFold {fold + 1}:\n##############")

        model = DebertaV2ForSequenceClassification.from_pretrained("../deberta-v3-base", num_labels=num_labels).to(device)

        # Split data into training and validation sets
        train_input_ids, train_attention_mask, train_labels = input_ids[train_indices], attention_mask[train_indices], labels[train_indices]
        val_input_ids, val_attention_mask, val_labels = input_ids[val_indices], attention_mask[val_indices], labels[val_indices]

        # Create DataLoader for training and validation
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
        best_sn, best_sp, best_acc, best_mcc, best_f1, best_auc, best_precision = 0, 0, 0, 0, 0, 0, 0
        min_difference = 1

        # Train the model for each fold
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

            # Evaluate on the validation set
            model.eval()
            val_predictions, val_probabilities, val_true_labels = [], [], []
            with torch.no_grad():
                for batch_input_ids, batch_attention_mask, batch_labels in val_loader:
                    outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
                    logits = outputs.logits
                    batch_preds = (logits >= 0.5).squeeze().cpu().numpy()
                    batch_probs = logits.cpu().detach().numpy()

                    val_predictions.extend(batch_preds.tolist())
                    val_probabilities.extend(batch_probs.tolist())
                    val_true_labels.extend(batch_labels.cpu().numpy())

            # Calculate metrics
            val_mcc = matthews_corrcoef(val_true_labels, val_predictions)
            val_auc = roc_auc_score(val_true_labels, val_probabilities)
            val_f1 = f1_score(val_true_labels, val_predictions)
            val_precision = precision_score(val_true_labels, val_predictions, zero_division=0)

            # Sensitivity and specificity
            TP = TN = FP = FN = 0
            for i in range(len(val_true_labels)):
                if val_predictions[i] == 1 and val_true_labels[i] == 1:
                    TP += 1
                elif val_predictions[i] == 0 and val_true_labels[i] == 0:
                    TN += 1
                elif val_predictions[i] == 1 and val_true_labels[i] == 0:
                    FP += 1
                else:
                    FN += 1

            val_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            val_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            val_accuracy = (TP + TN) / (TP + TN + FP + FN)

            print(f"Validation Sensitivity: {val_sensitivity:.4f} | Validation Specificity: {val_specificity:.4f} | "
                  f"Validation MCC: {val_mcc:.4f} | Validation Accuracy: {val_accuracy:.4f} | Validation Precision: {val_precision:.4f} | Validation F1: {val_f1:.4f} | "
                  f"Validation AUC: {val_auc:.4f}")

            # Store metrics
            current_difference = abs(val_sensitivity-val_specificity)
            if current_difference < min_difference:
                min_difference = current_difference
                best_auc = val_auc
                best_sn = val_sensitivity
                best_sp = val_specificity
                best_acc = val_accuracy
                best_mcc = val_mcc
                best_f1 = val_f1
                best_precision = val_precision
        print(f"Current:Best Validation Sensitivity: {best_sn:.4f} | Best Validation Specificity: {best_sp:.4f} | "
            f"Best Validation MCC: {best_mcc:.4f} | Best Validation Accuracy: {best_acc:.4f} | "
            f"Best Validation Precision: {best_precision:.4f} | Best Validation F1: {best_f1:.4f} | "
            f"Best Validation AUC: {best_auc:.4f}")
        sensitivity_list.append(best_sn)
        specificity_list.append(best_sp)
        accuracy_list.append(best_acc)
        mcc_list.append(best_mcc)
        f1_list.append(best_f1)
        auc_list.append(best_auc)
        precision_list.append(best_precision)

        # Write results to file
        with open("../Result/Result_output.txt", "a") as file:
            file.write(f"Fold {fold + 1} Results:\n")
            file.write(f"Sensitivity: {best_sn:.4f} | Specificity: {best_sp:.4f} | MCC: {best_mcc:.4f} | "
                    f"Accuracy: {best_acc:.4f} | Precision: {best_precision:.4f} | F1: {best_f1:.4f} | "
                    f"AUC: {best_auc:.4f}\n")

    # Cross-validation summary
    print("\nCross-Validation Results:")
    print(f"Sensitivity: {np.mean(sensitivity_list):.4f} | Specificity: {np.mean(specificity_list):.4f} "
        f"| MCC: {np.mean(mcc_list):.4f} | Accuracy: {np.mean(accuracy_list):.4f} | "
        f"Precision: {np.mean(precision_list):.4f} | F1: {np.mean(f1_list):.4f} | "
        f"AUC: {np.mean(auc_list):.4f}")

    with open("../Result/Result_output.txt", "a") as file:
        file.write("Cross-Validation Summary:\n")
        file.write(f"Sensitivity: {np.mean(sensitivity_list):.4f} | Specificity: {np.mean(specificity_list):.4f} "
                f"| MCC: {np.mean(mcc_list):.4f} | Accuracy: {np.mean(accuracy_list):.4f} | "
                f"Precision: {np.mean(precision_list):.4f} | F1: {np.mean(f1_list):.4f} | "
                f"AUC: {np.mean(auc_list):.4f}\n")
        file.write("-----------------------------------------\n")


    
    # # 模型完整训练
    # model = DebertaV2ForSequenceClassification.from_pretrained("../deberta-v3-base", num_labels=num_labels).to(device)
    # train_dataset = TensorDataset(input_ids, attention_mask, labels)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # # Optimizer and scheduler
    # optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    # total_steps = len(train_loader) * num_epochs
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    # model.train()
    # for epoch in range(num_epochs):
    #     total_loss = 0
    #     for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(batch_input_ids, batch_attention_mask, labels=batch_labels, return_dict=True)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         total_loss += loss.item()
    #     avg_loss = total_loss / len(train_loader)
    #     print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    # model_save = model.state_dict()
    # torch.save(model_save, "../Result/6T101_deberta_model.pth")


