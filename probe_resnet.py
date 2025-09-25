import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, resnet101, resnet152, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.transforms import CenterCrop, Normalize, Resize
import torch.func as func
from matplotlib import pyplot as plt 
from diffusers.models import AutoencoderKL, AutoencoderTiny
from typing import Union, Tuple
import langevin
from datasets import load_dataset

Tensor = torch.Tensor

device = 'mps' #or 'cpu'

resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
def transform(x:Tensor):
    x = torch.nn.functional.interpolate(x, size=256, mode='bilinear', align_corners=False)
    x = x.squeeze(0)
    x = CenterCrop(224)(x)
    
    # normalize, but makes sure it is safe among devices
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    x = (x - mean) / std

    return x.unsqueeze(0)

#vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
vae.to(device=device)
vae.eval()
#vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
gen = vae.decoder
gen.eval()
# gan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
#                      model_name = 'celebAHQ-256', pretrained = True)

# gangen : nn.Module = gan.netG
# gangen.eval()
# resnet.eval()

# gangen.to(device = device)
resnet.eval()
resnet.to(device = device)

#z0 = torch.randn([1, 512], device = device, requires_grad=True)
z0 = torch.randn([1,16,4,4], device = device, requires_grad=True)
#z0 = torch.randn([1,4,4,4], device = device, requires_grad=True)

def G_function(
    model       : nn.Module,
    generator   : nn.Module,
    beta        : float,
    l1_reg      : float,
    label       : int,
    anchor_z    : Union[Tensor, float] = 0. # float will be broadcast 
):
    model.eval()
    generator.eval()
    criterion = nn.CrossEntropyLoss()
    tensor_label = torch.tensor([label], device = device)
    def G(z: Tensor) -> Tensor:
        noise = torch.randn_like(z)*0.05
        zprime = z + noise
        x = generator(zprime)
        x = torch.clamp(x, 0, 1)
        logits = model(transform(x))
        loss = criterion(logits, tensor_label)
        dist = torch.mean((zprime - anchor_z).abs())
        rce = reconst_error(zprime)
        return beta*(loss + l1_reg*dist + rce)

    return G

def differentiate(G, retain_graph=True):
    """Returns a gradient function that retains the graph."""
    def grad_fn(z):
        loss = G(z)                          # Compute loss
        grad = torch.autograd.grad(
            loss, z, 
            retain_graph=retain_graph,        # <-- Critical for reuse
            create_graph=False                # Disable if not needed
        )[0]
        return grad
    return grad_fn

def show_image(z :Tensor, save_filename = None):
    x :Tensor  = gen(z)
    x = x[0].detach().cpu().numpy().transpose([1,2,0])
    x = x.clip(0,1)
    plt.imshow(x)
    if save_filename is not None:
        plt.savefig(save_filename, bbox_inches = 'tight')
    plt.show()
    

def resize_latent(
        z : Tensor, # latent variable 
        m : int     # size of the new latent variable
        )->Tensor:
    """
    Resizes the image, and using the encoder finds 
    the corresponding latent variable which is compressed 8 times.
    """
    out = gen(z)
    img = torch.clip(out, 0,1)
    img = Resize(8*m)(img.cpu()).to(device)
    return vae.encoder(img)

def probabilities(z: Tensor):
    out = gen(z)
    img = torch.clip(out, 0,1)
    logits = resnet(transform(img))
    probs = torch.softmax(logits,1)
    return probs[0]

def reconst_error(z: Tensor):
    x = torch.clip(vae.decoder(z),0,1)
    x = Resize(256)(x.cpu()).to(device)
    xprime = vae.decoder(vae.encoder(x))
    return nn.functional.mse_loss(x,xprime)

def search_loop(
    z               : Tensor,
    iter_n          : int,      # number of langevin steps before checking
    prob_threshold  : float,    # the criterion to get out of langevin loop
    beta            : float,
    step_size       : float,
    l1_reg          : float,
    label           : int,
    anchor          : Union[Tensor, float]
)-> Tensor:
    print(f"beta:{beta}, step_size:{step_size}")
    prob = 0.
    while prob < prob_threshold:
        funcG = G_function(resnet, gen, beta, l1_reg, label, anchor)
        gradG = differentiate(funcG)
        etaG = step_size/beta
        hypsG = funcG, gradG, etaG, -4,4
        print(f"step_size: {step_size:.4f}, beta: {beta:.2f}, l2_reg:{l1_reg:.2f}")
        for i in range(iter_n):
            z = langevin.torch_MALA_step(z, hypsG)
            if i%10==9:
                print(f"G = {funcG(z).item()/beta:.4f}, \t reconst {reconst_error(z).item():.4f},\
                anchor_dist {torch.mean((z - anchor).pow(2)).item():.4f}")
        prob = probabilities(z)[label].item()
        print("probability: ", prob)
        step_size = step_size*0.9
        l1_reg = l1_reg*0.9
        beta = beta*1.1
    return z


def main(label = 1):
    #sizes       = [   4,      8,      16,     32]
    sizes        = [ 5,  8,  12,  24,  32]
    #prob_crits  = [  .6,     .7,     .8,     .9]
    prob_crits   = [ .7, .8, .9, .9, .9]
    #betas       = [  20,     40,      80,    160]
    betas        = [  50, 100, 200, 400, 800]
    #step_sizes  = [  .2,     .1,    .05,    .025]
    step_sizes   = [ .2, .1, 0.05, 0.02, 0.01]
    #l1_regs     = [  .00,    .1,   .5,    1.]
    l1_regs      = [ .0, 0.1, 1., 1.5, 2.]
    z = torch.randn(
        [1,16,sizes[0],sizes[0]], 
        device = device, 
        requires_grad=True
        )           # latent variable
    zs= []
    ps = []
    anchor = 0.
    for i, size in enumerate(sizes):
        z = resize_latent(z, size)
        anchor = z.clone()
        print(f"Size of image {(size*8, size*8)}")
        show_image(z)
        prob_thresh = prob_crits[i]
        beta = betas[i]
        step_size = step_sizes[i]
        l1_reg = l1_regs[i]
        z :Tensor = search_loop(z, 100, prob_thresh, beta, step_size, l1_reg, label, anchor)
        zs.append(z)
        probs = probabilities(z)
        print("high prob classes: ",torch.where(probs>0.05)[0])
        prob = probs[label].item()
        ps.append(prob)
        show_image(z)
    return zs, ps


