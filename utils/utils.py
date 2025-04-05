import os
import random

import numpy as np
import torch


def save_ckpt(model_name, model, optimizer, scheduler, ckpt_folder):
    """
    save checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
        },
        os.path.join(ckpt_folder, f'{model_name}.pth')
    )


def load_ckpt(model, model_name, ckpt_folder, device):
    """
    load checkpoint
    """
    if not os.path.exists(ckpt_folder):
        raise ValueError(f"Checkpoint folder {ckpt_folder} does not exist.")
    checkpoint = torch.load(os.path.join(ckpt_folder, f'{model_name}.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)