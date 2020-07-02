import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import sys
from collections import OrderedDict
import torchvision.models as models
from PIL import Image


def load_model(weight_path):
    weights = torch.load(weight_path)
    resnet = models.resnet18(pretrained=False)
    names = [n for n in resnet._modules.keys()]
    model = nn.Sequential(
        OrderedDict(
            [
                (names[i], n)
                for i, n in enumerate(list(resnet.children())[:-1])
            ]
        )
    )
    model.load_state_dict(weights)
    model.eval()
    return model


if __name__ == '__main__':
    model = load_model(sys.argv[1])
    df_lis = pd.read_csv(sys.argv[2]).to_dict(orient="records")[:10]
    out_path = sys.argv[3]
    writer = SummaryWriter()

    # transfer
    data_transforms = transforms.Compose([transforms.ToTensor()])
    embeddings = []
    images = []
    with torch.no_grad():
        model.eval()
        for n in df_lis:
            img = Image.open(n['img_path'])
            img = img.resize((256, 256))
            img = data_transforms(img)
            img = img.unsqueeze(0)
            images.append(img)
            out = model(img).reshape(1, 512)
            embeddings.append(out)
    final_embeddings = np.concatenate(embeddings)
    final_images = np.concatenate(images)
    writer.add_embedding(final_embeddings, label_img=final_images)
