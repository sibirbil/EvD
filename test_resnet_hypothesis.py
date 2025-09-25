import torch
from typing import Union, Tuple, Callable, List
from datasets import load_dataset
from torchvision import transforms

from probe_resnet import resnet, gen
Tensor = torch.Tensor

ds = load_dataset("ILSVRC/imagenet-1k", split = "validation")

preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

preprocess_unnormalize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def get_tensorlist(
        label :int,
        process : Callable = preprocess,
        device = 'mps'
        ):

    sub_ds = ds.filter(lambda example: example['label']== label)
    tensorlist = []
    for sample in sub_ds:
        pil_image = sample['image']
        tensor :Tensor = process(pil_image)
        tensorlist.append(tensor.to(device = device)) 
    
    return tensorlist


resnet.eval()

def modify(
    tensorlist : List[Tensor],
    modification : Union[str, Tuple[int, int, int]],
    normalized : bool = True
    )-> List[Tensor]:
    imgs_batch = torch.stack(tensorlist, dim = 0)
    bw_batch = channel_permutation_transform(imgs_batch, modification, normalized = normalized)
    return [bw_img.clone() for bw_img in bw_batch]
    


def modified_performance(label, modification = 'grayscale'):
    
    imgs_tl = get_tensorlist(label, preprocess)
    print(f"imgs_tl length and shape {len(imgs_tl), imgs_tl[0].shape}")
    imgs_batch = torch.stack(imgs_tl, dim =0)
    print(f"imgs_batch shape {imgs_batch.shape}")
    logits = resnet(imgs_batch)
    preds = torch.argmax(logits, dim  = 1)
    probs = torch.softmax(logits, dim = 1)
    pred_probs = probs[:, label]

    # turn the images to black and white
    bw_imgs_batch = channel_permutation_transform(imgs_batch, modification)
    print(f"bw_imgs_batch, shape:{bw_imgs_batch.shape}, max:{bw_imgs_batch.max():.4f}, min:{bw_imgs_batch.min():.4f}")

    bw_logits = resnet(bw_imgs_batch)
    bw_preds = torch.argmax(bw_logits, dim = 1)
    bw_probs = torch.softmax(bw_logits, dim = 1)
    bw_pred_probs = bw_probs[:, label]

    return preds, pred_probs, bw_preds, bw_pred_probs

def channel_permutation_transform(
        tensor: torch.Tensor, 
        permutation: Union[Tuple[int, int, int], str] = (2, 1, 0),
        normalized : bool = True     #whether the data comes in imagenet normalized
    ) -> torch.Tensor:
    """
    Apply channel permutation transformations to RGB images using tuple notation.
    
    Args:
        tensor: Input tensor of shape [C, H, W] or [B, C, H, W] where C=3 (RGB)
        permutation: Either a tuple (a, b, c) where:
            - Channel 0 (Red) goes to position a
            - Channel 1 (Green) goes to position b  
            - Channel 2 (Blue) goes to position c
            Or a string shorthand:
            - 'identity': (0, 1, 2) - No change
            - 'rb_swap': (2, 1, 0) - Red ↔ Blue
            - 'rg_swap': (1, 0, 2) - Red ↔ Green
            - 'gb_swap': (0, 2, 1) - Green ↔ Blue
            - 'rgb_to_brg': (1, 2, 0) - Circular shift: RGB → BRG
            - 'rgb_to_gbr': (2, 0, 1) - Circular shift: RGB → GBR
            - 'invert_channels': Special case for channel inversion
            - 'grayscale': Special case for grayscale conversion
    
    Returns:
        Transformed tensor with same shape as input
        
    Examples:
        # Red-Blue swap: (R,G,B) → (B,G,R)
        result = channel_permutation_transform(tensor, (2, 1, 0))
        
        # Green-Blue swap: (R,G,B) → (R,B,G)  
        result = channel_permutation_transform(tensor, (0, 2, 1))
        
        # Circular shift: (R,G,B) → (B,R,G)
        result = channel_permutation_transform(tensor, (1, 2, 0))
    """
    if tensor.dim() not in [3, 4]:
        raise ValueError("Tensor must be 3D [C,H,W] or 4D [B,C,H,W]")
    
    # Handle both batched and single image tensors
    original_shape = tensor.shape
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    batch_size, channels, height, width = tensor.shape
    
    if channels != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {channels}")
    
    # Convert string shortcuts to tuples
    if isinstance(permutation, str):
        permutation_map = {
            'identity': (0, 1, 2),
            'rb_swap': (2, 1, 0),
            'rg_swap': (1, 0, 2), 
            'gb_swap': (0, 2, 1),
            'rgb_to_brg': (1, 2, 0),
            'rgb_to_gbr': (2, 0, 1),
        }
        
        if permutation in permutation_map:
            permutation = permutation_map[permutation]
        elif permutation == 'invert_channels':
            return _invert_channels(tensor, normalized)
        elif permutation == 'grayscale':
            return _convert_to_grayscale(tensor, normalized)
        else:
            raise ValueError(f"Unknown permutation string: {permutation}")
    
    # Validate permutation tuple
    if (len(permutation) != 3 or 
        set(permutation) != {0, 1, 2} or
        not all(isinstance(x, int) for x in permutation)):
        raise ValueError("Permutation must be a tuple of (0,1,2) in some order")
    
    # Apply permutation
    result = torch.zeros_like(tensor)
    for source_channel, target_channel in enumerate(permutation):
        result[:, target_channel] = tensor[:, source_channel]
    
    # Return to original shape
    if len(original_shape) == 3:
        result = result.squeeze(0)
    
    return result

def _invert_channels(
        image: Tensor, 
        normalized : bool = True) -> Tensor:
    """Helper function to invert channels (1 - x for each channel)."""
    # Check if tensor is normalized (ImageNet style)
    if normalized:  # Likely normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        # Denormalize
        denorm = image * std + mean
        # Invert
        inverted = 1.0 - denorm
        # Renormalize
        result = (inverted - mean) / std
    else:
        assert image.min()> -1e-6, "probably image is normalized, set normalized = True"
        result = 1.0 - image

    
    return result


def _convert_to_grayscale(
        image: Tensor, 
        normalized : bool = True
    ) -> Tensor:
    """Helper function to convert to grayscale."""
    if normalized:  # unnormalize, apply luminance formula and go back
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        # Denormalize
        denorm = image * std + mean
        # Convert to grayscale using luminance formula: 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * denorm[:, 0] + 0.587 * denorm[:, 1] + 0.114 * denorm[:, 2]
        # Replicate grayscale to all 3 channels
        result = gray.unsqueeze(1).repeat(1, 3, 1, 1)
        # renormalize
        result = (result - mean)/std
    
    else:
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        result = gray.unsqueeze(1).repeat(1,3,1,1)
    
    return result

import math

def main(index_list = None):
    names =  ds.info.features['label'].names
    if index_list is None:
        index_list = [1, 113, 345, 937, 943]
    modlist = ['grayscale', 'invert_channels', 'rg_swap', 'rb_swap', 'gb_swap']
    results = []
    for idx in index_list:
        name = names[idx]
        for modification in modlist:
            results.append({'name':name, 'mod':modification})
            preds, probs, mod_preds, mod_probs = modified_performance(idx, modification)
            print(name, '\n', preds, '\n', mod_preds)
            perf = (preds==idx).sum()
            modperf = (mod_preds == idx).sum()
            probdiff_mean = (probs - mod_probs).mean() 
            probdiff_ste = (probs - mod_probs).std()/math.sqrt(50)
            results[-1]['orig_perf']        = perf.item()
            results[-1]['mod_perf']         = modperf.item()
            results[-1]['prob_diff_mean']   = probdiff_mean.item()
            results[-1]['prob_diff_ste']    = probdiff_ste.item()
    return results