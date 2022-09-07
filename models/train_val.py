import sys
sys.path.append("../scop_classification_by_ESM")

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import SCOPDataset
import Model
from config import Config

# hyperparameters
config = Config()
out_filename = config.get_model_name()
print(out_filename)

# all_data_file_path="data/splits/debug/all_cleaned.txt"
# train_data_file_path="data/splits/debug/train_70.txt"
# val_data_file_path="data/splits/debug/val_14.txt"

all_data_file_path="data/splits/all_cleaned.txt"
train_data_file_path="data/splits/train_24538.txt"
val_data_file_path="data/splits/val_4458.txt"

class_dict, n_classes = SCOPDataset.generate_class_dict(all_data_file_path, config.task)
class_weights = SCOPDataset.compute_class_weights(train_data_file_path, config.task, config.device)


model = Model.ESMClassifier(n_classes).to(config.device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.init_lr)
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")



# # dataset and dataloader
train_dataset = SCOPDataset.X(train_data_file_path, model.batch_converter, class_dict, config.task, config.max_len)
val_dataset = SCOPDataset.X(val_data_file_path, model.batch_converter, class_dict, config.task, config.max_len)
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False)
print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

best_loss = torch.inf
for epoch in range(config.n_epochs):
    train_loss = Model.train(model, optimizer, criterion, train_loader, config.device)
    # print(f"{epoch}/{n_epochs}: train_loss={train_loss:.4f}")

    if epoch%10 != 0: continue
    
    val_loss, true_scores, pred_scores = Model.val(model, criterion, val_loader, config.device)
    metrics = Model.compute_clssification_metrics(true_scores, pred_scores.argmax(axis=1))

    print(f"{epoch}/{config.n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={metrics['acc']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")

    writer.add_scalar('train loss',train_loss,epoch)
    writer.add_scalar('val loss',val_loss,epoch)
    writer.add_scalar('acc',metrics["acc"],epoch)

    # save model dict
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}.pth")
                    
    # break
