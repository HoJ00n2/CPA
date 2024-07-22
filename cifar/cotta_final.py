from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
import robusta

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
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


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        # doesn't use model_anchor
        # self.model_state, self.optimizer_state, self.model_ema = \
        #     copy_model_and_optimizer(self.model, self.optimizer)
        
        self.transform = get_tta_transforms()    
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.clf = Classifier().cuda() # classifier must be on cuda!

        round, cnt, sum = 0, 0, 0
        self.cnt = cnt
        self.round = round
        self.sum = sum
        mean = 0
        self.mean = mean
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
        # Use this line to also restore the teacher model  >> for default setting                       
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        # our setting use this code
        # self.model_state, self.optimizer_state, self.model_ema = \
        #     copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        outputs = model(x)[0] # self.model(x) >> model(x)[0]
        # Teacher Prediction
        threshold = torch.nn.functional.softmax(self.model_anchor(x)[0], dim=1).max(1)[0]
        anchor_prob = torch.nn.functional.softmax(outputs, dim=1).max(1)[0] # self.model_anchor(x)[0]
        standard_ema = self.model_ema(x)[0]
        # Augmentation-averaged Prediction
        N = 32 
        outputs_emas = []
        
        # Threshold choice discussed in supplementary
        # 처음엔 고정값과 비교
        if anchor_prob.mean(0) < 0.88 + 0.01*threshold.mean(0): # self.ap 0.88~89 for cifar10 , 0.75 for cifar100
        #if threshold.mean(0) < anchor_prob.mean(0) + 0.05:
            for _ in range(N):
                outputs_  = self.model_ema(self.transform(x))[0].detach()
                outputs_emas.append(outputs_)
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
    
        # to get class-wise centroid 
        #t_clf = self.clf(self.model_ema(x)[1]) # classifier에 넣어 [640, 10] 의 matrix 얻기
        #t_gen = self.model_ema(x)[1] # fc weight [640, 10] >> (x)[1] >> [200, 640]
        #s_c = self.model_anchor.fc.weight # [10,640] for cifar10 
        #t_c = update_centers(t_gen, t_clf)

        # regularize (MI loss)
        softmax_output = torch.softmax(outputs, dim=1) # original is student outputs       choice : outputs_ema, outputs, mean_output  >> (outputs_ema & outputs)   or  (outputs_student & standard_ema) setting 
        margin_output = torch.mean(softmax_output, dim=0)
        log_softmax_output = torch.log_softmax(outputs, dim=1) # choice : (outputs_ema & outputs)   or  (outputs_student & standard_ema) setting        
        log_margin_output = torch.log(margin_output + 1e-5) # 1e-5
        mutual_info_loss = -1 * torch.mean(
            torch.sum(softmax_output * (log_softmax_output - log_margin_output), dim=1)
        )

        # Student update 
        # CE + MI + MSTN
        #center_loss = torch.nn.MSELoss(reduction='mean') 
        #C_loss = center_loss(s_c, t_c)#.requires_grad_=True
        
        loss = (softmax_entropy(outputs, outputs_ema)).mean(0) + mutual_info_loss #+ C_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)
        
        # Stochastic restore  # always > conditional by self.ap
        #if True:
        # cifar10 is fit at 0.88 ~ 0.89 
        # cifar100 is fit at near 0.75 (roughly)
        if anchor_prob.mean(0) < 0.88 + 0.01*threshold.mean(0): # - 0.01: #0.86:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() # self.rst = 0.01
                        #print(mask)
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * (mask) + p * (1.-mask)

        return outputs_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1) 

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.main = nn.Sequential( # nn.Linear(n_features, n_class) input x shape should be [200, 640]
            nn.Linear(640, 10), # origin (10,10) > (640, 10) error because input x shape is [200, 10] 
            nn.Softmax(dim=1) 
        )
    
    def forward(self, x):
        #print(self.main(x).is_cuda)
        return self.main(x)

def one_hot(batch, classes):
    ones = torch.eye(classes).cuda() # add .cuda()
    return ones.index_select(0,batch)

def update_centers(t_gen, t_clf): 
    tgt_1 = torch.argmax(t_clf,1).reshape(t_clf.size(0),1).detach()
    t_center = torch.zeros(10, 640, device=torch.device('cuda')) # for cifar 10
    t_zeros = torch.zeros(tgt_1.size()[1:], device=torch.device('cuda'))
    
    for i in range(10): # for cifiar 10 > range(10), for cifiar 100 > range(100)
        t_cur = torch.where(tgt_1.eq(i),t_gen, t_zeros).mean(0)  # CoTTA (src_1)[200,1] x (s_gen)[10, 640] >> must (batch, feature)
        t_center[i] = t_cur * (1 - 0.7) + t_center[i]*0.7        # t_center * 0.3
       
    return t_center


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


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor
    #return model_state, optimizer_state, ema_model


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
