import torch.hub
import torch.nn as nn

def fc_bc_relu_dropout(in_size, out_size, p=0.5):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.BatchNorm1d(out_size),
        nn.ReLU(),
        nn.Dropout(p=p),
    )


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        print(m)


def count_parameters(model):
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_p, non_trainable_p


def get_train_shuffleNet_v2_model(freeze=False):
    # shufflenet_v2_x0_5 = torch.hub.load('pytorch/vision:v0.9.1', 'shufflenet_v2_x0_5', pretrained=True)
    shufflenet_v2_x0_5 = torch.hub.load('./vision-0.9.1', 'shufflenet_v2_x0_5', source='local', pretrained=False)
    shufflenet_v2_x0_5.load_state_dict(torch.load("./shufflenetv2_x0.5-f707e7126e.pth"))

    fc_model = nn.Sequential(
        nn.Dropout(p=0.5),
        fc_bc_relu_dropout(1000, 256, 0.5),
        fc_bc_relu_dropout(256, 128, 0.5),
        nn.Linear(128, 1),    
        nn.Sigmoid(),
    )
    
    print('init weights [nn.Linear] layers:')
    fc_model.apply(weights_init)

    model = nn.Sequential(
        shufflenet_v2_x0_5,
        fc_model,
    )

    # freeze or not freeze pretrain weight
    if freeze:        
        for p in model[0].parameters():
            p.requires_grad = False # freeze
    else:
        for p in model[0].parameters():
            p.requires_grad = True # not freeze
    

    print('# of parameters')
    trainable_p, non_trainable_p = count_parameters(model)
    print('trainable:     %15s' % format(trainable_p, ','))
    print('non-trainable: %15s' % (format(non_trainable_p, ',')))
        
    return model


def get_eval_shuffleNet_v2_model(device):
    # shufflenet_v2_x0_5 = torch.hub.load('pytorch/vision:v0.9.1', 'shufflenet_v2_x0_5', pretrained=True)
    shufflenet_v2_x0_5 = torch.hub.load('./vision-0.9.1', 'shufflenet_v2_x0_5', source='local', pretrained=False)
    shufflenet_v2_x0_5.load_state_dict(torch.load("./shufflenetv2_x0.5-f707e7126e.pth"))

    fc_model = nn.Sequential(
        nn.Dropout(p=0.5),
        fc_bc_relu_dropout(1000, 256, 0.5),
        fc_bc_relu_dropout(256, 128, 0.5),
        nn.Linear(128, 1),    
        nn.Sigmoid(),
    )
    
    model = nn.Sequential(
        shufflenet_v2_x0_5,
        fc_model,
    )

    device = torch.device(device)
    model.to(device)
    model.eval()
    return model


class ParallelWearingModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        # M個clf, N個yolo rectangle
        # 輸出要是 M*N
        
        result = []
        for model in self.models:
            result.append(model(x))
        
        try:
            result = torch.reshape(torch.cat(result), (len(result), -1)).tolist()
        except Exception as e:
            print("torch.cat() function cannot concatenate empty tensors")

        return result


