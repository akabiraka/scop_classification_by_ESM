import sys
sys.path.append("../scop_classification_by_ESM")

import torch
import esm
import numpy as np

class ESMClassifier(torch.nn.Module):
    def __init__(self, n_classes) -> None:
        super(ESMClassifier, self).__init__()

        self.model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()
        # model.eval()  # disables dropout for deterministic results

        self.classifier = torch.nn.Linear(768, n_classes)


    def forward(self, batch_seq_tokens, batch_seq_lengths): 
        # batch_seq_tokens: [batch_size, max_len]
        # batch_seq_lengths: [batch_size]
        # print(batch_seq_lengths)
        # batch_seq_lengths = batch_seq_lengths.detach().cpu().numpy().tolist()

        # Extract per-residue representations (on CPU)
        # with torch.no_grad():
        results = self.model(batch_seq_tokens, repr_layers=[12], return_contacts=False)
        token_representations = results["representations"][12]


        ## Generate per-sequence representations via averaging
        ## NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, seq_len in enumerate(batch_seq_lengths):
            sequence_representations.append(token_representations[i, 1 : seq_len.item()+1].mean(0))

        sequence_representations = torch.stack(sequence_representations) # shape: [batch_size, 768]
        # print(sequence_representations.shape)

        out = self.classifier(sequence_representations)
        return out


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
def compute_clssification_metrics(target_classes, pred_classes):
    acc = accuracy_score(target_classes, pred_classes)
    precision = precision_score(target_classes, pred_classes, average="weighted", zero_division=1)
    recall = recall_score(target_classes, pred_classes, average="weighted", zero_division=1)
    f1 = f1_score(target_classes, pred_classes, average="weighted", zero_division=1)
    return {"acc": acc, 
            "precision": precision, 
            "recall": recall, 
            "f1": f1,
            "pred_classes": pred_classes, 
            "target_classes": target_classes}

def get_one_hot(labels, n_classes):
    # labels shape: (n,)
    shape = (labels.size, n_classes)
    one_hot = np.zeros(shape)
    rows = np.arange(labels.size)
    one_hot[rows, labels] = 1
    return one_hot 

from sklearn.metrics import roc_auc_score
def compute_roc_auc_score(target_classes, pred_cls_distributions, n_classes):
    target_cls_distributions = get_one_hot(target_classes, n_classes)
    roc_auc = roc_auc_score(target_cls_distributions, pred_cls_distributions, average="samples", multi_class="ovr")
    return roc_auc

@torch.no_grad()
def val(model, criterion, data_loader, device): 
    print("validating ...")
    model.eval()
    pred_scores, true_scores = [], []
    total_loss = 0.0
    for i, (seqs_tokens, y_true, seqs_lens) in enumerate(data_loader):
        # print(seqs_tokens.shape, y_true.shape, seqs_lens.shape)
        seqs_tokens, y_true, seqs_lens = seqs_tokens.to(device), y_true.to(device), seqs_lens.to(device)
        model.zero_grad(set_to_none=True)
        y_pred = model(seqs_tokens, seqs_lens)
        print(y_pred.shape)

        loss = criterion(y_pred, y_true)
        total_loss += loss.item()
        # print(f"    batch no: {i}, loss: {loss.item()}")
        
        pred_scores.append(torch.nn.functional.softmax(y_pred, dim=1).detach().cpu().numpy())
        true_scores.append(y_true.detach().cpu().numpy())
        
        # if i==5: break

    
    true_scores, pred_scores = np.hstack(true_scores), np.vstack(pred_scores) #true_scores: [1, batch_size], pred_scores: [batch_size, n_classes]
    print(true_scores.shape, pred_scores.shape)
    return total_loss/len(data_loader), true_scores, pred_scores


# pred_labels.append(y_pred.argmax(dim=1).cpu().numpy())
# true_labels.append(y_true.cpu().numpy())


def train(model, optimizer, criterion, data_loader, device): 
    print("training ...")
    model.train()
    total_loss = 0.0
    for i, (seqs_tokens, y_true, seqs_lens) in enumerate(data_loader):
        # print(seqs_tokens.shape, y_true.shape, seqs_lens.shape)
        seqs_tokens, y_true, seqs_lens = seqs_tokens.to(device), y_true.to(device), seqs_lens.to(device)
        model.zero_grad(set_to_none=True)
        y_pred = model(seqs_tokens, seqs_lens)
        #print(y_pred.shape, y_pred.device, y_true.shape, y_true.device)

        loss = criterion(y_pred, y_true)
        total_loss += loss.item()
        
        
        loss.backward()
        optimizer.step()        
        
        # print(f"    batch no: {i}, loss: {loss.item()}")
        # break

    return total_loss/len(data_loader)
    
    
