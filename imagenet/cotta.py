from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging

import numpy as np

logger = logging.getLogger(__name__)

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()                   
        
        round, cnt = 0, 0
        self.cnt = cnt
        self.round = round
        mean = 0
        self.mean = mean                                        
        self.clf = Classifier().cuda()

    def forward(self, x):                                       
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also 
        # want to del self.model_ema first to save gpu memory.
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)                         


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer): # x [64, 3, 224, 224] >> batchsize is 64
        #robusta.batchnorm.adapt(self.model, adapt_type="batch_wise")
        #robusta.batchnorm.adapt(self.model_ema, adapt_type="batch_wise")
        outputs = self.model(x, stage=None) # [64, 1000]
        #self.model_ema.train()
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(outputs, dim=1).max(1)[0] # shape [64, 1000]
        anchor_pred = torch.nn.functional.softmax(outputs, dim=1).max(1)[1]
        standard_ema = self.model_ema(x,stage=None)
        teacher_prob = np.array(torch.nn.functional.softmax(standard_ema, dim=1).max(1)[0].cpu())
        teacher_pred = np.array(torch.nn.functional.softmax(standard_ema, dim=1).max(1)[1].cpu())

        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        am = anchor_prob.mean(0) 
        
        # adaptive threshold
        cls_thresh = IAST(1000, teacher_prob, teacher_pred) # [class num]
        
        # for imagenet anchor_mean =0.72
        #logger.info(f"anchor mean : {am : .2%}")
        # if am < 0.3 :  # ours 0.3
        #     # self.cnt += 1
        #     # print(self.cnt)
        #     for _ in range(N):
        #         outputs_  = self.model_ema(self.transform(x),stage=None).detach()
        #         outputs_emas.append(outputs_)
        #     outputs_ema = torch.stack(outputs_emas).mean(0)

        for _ in range(N):
            outputs_  = self.model_ema(self.transform(x),stage=None).detach()
            outputs_emas.append(outputs_)
            outputs_ema = torch.stack(outputs_emas).mean(0)

        # Threshold choice discussed in supplementary
        # if to_aug:
        #     outputs_ema = torch.stack(outputs_emas).mean(0)
        # else:
        #     outputs_ema = standard_ema

        for i in range(1000) :
            index = torch.where(anchor_pred==i)[0]
            tmp_prob = anchor_prob[index]
                
            if tmp_prob.mean(0) > cls_thresh[i]: 
                outputs_ema[index] = standard_ema[index]

        # Student update

        # MSTN
        #t_clf = self.clf(self.model_ema(x)) # self.model_ema(x)[1] >> [64, 1000] = [batch, class]
        #t_gen = self.model_ema(x) # self.model_ema(x)[1],  [64, 1000]
        #s_c = self.model_anchor.head.fc.weight # [1000,2048] for imagenet
        #t_c = self.model_ema.head.fc.weight # [1000,2048]

        # regularize (MI loss)
        softmax_output = torch.softmax(outputs, dim=1) # original is student outputs       choice : outputs_ema, outputs, mean_output  >> (outputs_ema & outputs)   or  (outputs_student & standard_ema) setting 
        margin_output = torch.mean(softmax_output, dim=0)
        log_softmax_output = torch.log_softmax(outputs, dim=1) # choice : (outputs_ema & outputs)   or  (outputs_student & standard_ema) setting        
        log_margin_output = torch.log(margin_output + 1e-5) # 1e-5
        mutual_info_loss = -1 * torch.mean(
            torch.sum(softmax_output * (log_softmax_output - log_margin_output), dim=1)
        )
        # original loss (+ IM)
        #loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0) + mutual_info_loss

        # CE + IM + MSTN
        #center_loss = torch.nn.MSELoss(reduction='mean') 
        #C_loss = center_loss(s_c, t_c)#.requires_grad_=True
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) + mutual_info_loss #+ C_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999) # alpha = 0.999
        
        # Stochastic restore
        if True:
        #if am < 0.3 :
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.001).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return outputs_ema

###############################################################################
def IAST(n, prob, pred): # probs_: [class,batch]
    #cls_thresh = np.empty(n) # [class,batch]
    cls_thresh = np.ones(n)*0.5

    logits_cls_dict = {c: [cls_thresh[c]] for c in range(n)} # [class]
    for cls in range(n):
        logits_cls_dict[cls].extend(prob[pred == cls].astype(np.float16))
            
    # instance adaptive selector
    alpha, beta, gamma = 0.2, 0.9, 8.0 # 0.2, 0.5, 8.0
    # cls_thresh = ias_thresh(logits_cls_dict, alpha=alpha, n=n, w=cls_thresh, gamma=gamma) 
    tmp_cls_thresh =  ias_thresh(logits_cls_dict, alpha=alpha, n=n, w=cls_thresh, gamma=gamma) 
    cls_thresh = beta*cls_thresh + (1-beta)*tmp_cls_thresh 
    cls_thresh[cls_thresh>=1] = 0.5

    return cls_thresh

def ias_thresh(conf_dict, alpha, n=10, w=None, gamma=1.0):
    if w is None:
        w = np.ones(n)
    # threshold 
    cls_thresh = np.ones(n, dtype = np.float32)
    for idx_cls in np.arange(0, n):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh
##############################################################################

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    #print(nm, np)
    return params, names

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.main = nn.Sequential( # nn.Linear(n_features, n_class) input x shape should be [200, 640]
            nn.Linear(64,1000), # for imagenet
            nn.Softmax(dim=1) 
        )
    
    def forward(self, x):
        # x.shape is [64, 1000] [batch, class]
        return self.main(x) # return is [batch, class] >> x is already fit! >> no need classifier

def one_hot(batch, classes):
    ones = torch.eye(classes).cuda() # add .cuda()
    return ones.index_select(0,batch)

def update_centers(t_gen, t_clf): 
    tgt_1 = torch.argmax(t_clf,1).reshape(t_clf.size(0),1).detach()
    t_center = torch.zeros(1000,2048, device=torch.device('cuda')) # for imagenet
    t_zeros = torch.zeros(tgt_1.size()[1:], device=torch.device('cuda'))
    
    for i in range(1000): # for cifiar 10 > range(10),   for cifiar 100 > range(100)
        t_cur = torch.where(tgt_1.eq(i),t_gen, t_zeros).mean(0)  # CoTTA (src_1)[200,1] x (s_gen)[10, 640] >> must (batch, feature)
        t_center[i] = t_cur * (1 - 0.7) + t_center[i]*0.7        # t_center * 0.3
       
    return t_center


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
