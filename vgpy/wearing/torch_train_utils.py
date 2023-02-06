import torch

class EarlyStopping():
    # https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
#             print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class History:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.running_loss = 0
        self.running_acc = 0
        
    def batch_train(self, batch_loss, batch_acc):
        self.running_loss += batch_loss
        self.running_acc += batch_acc
        
    def epoch_train(self, batch_num):
        self.train_loss.append(self.running_loss / batch_num)
        self.train_acc.append(self.running_acc / batch_num)
        self.running_loss = 0
        self.running_acc = 0
        
    def epoch_val(self, val_loss, val_acc):
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)


def binary_acc(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum/y_true.shape[0]
#     acc = torch.round(acc * 100)
    
    return acc



