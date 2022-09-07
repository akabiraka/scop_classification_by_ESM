import sys
sys.path.append("../scop_classification_by_ESM")

import torch
from torch.utils.data import DataLoader

import SCOPDataset
import Model
from config import Config

# hyperparameters
config = Config()
out_filename = config.get_model_name()
print(out_filename)

all_data_file_path="data/splits/all_cleaned.txt"
train_data_file_path="data/splits/train_24538.txt"
test_data_file_path="data/splits/test_5862.txt"

class_dict, n_classes = SCOPDataset.generate_class_dict(all_data_file_path, config.task)
class_weights = SCOPDataset.compute_class_weights(train_data_file_path, config.task, config.device)

# loading model and best checkpoint
model = Model.ESMClassifier(n_classes).to(config.device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])


# dataset and dataloader
test_dataset = SCOPDataset.X(test_data_file_path, model.batch_converter, class_dict, config.task, config.max_len)
test_loader = DataLoader(test_dataset, config.batch_size, shuffle=False)
print(f"test batches: {len(test_loader)}")


val_loss, true_scores, pred_scores = Model.val(model, criterion, test_loader, config.device)
metrics = Model.compute_clssification_metrics(true_scores, pred_scores.argmax(axis=1))
#roc_auc = Model.compute_roc_auc_score(true_scores, pred_scores, n_classes)

print(f"acc={metrics['acc']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")#, roc_auc={roc_auc:.4f}")
