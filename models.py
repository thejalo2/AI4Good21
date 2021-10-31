import timm
import torch
import torch.nn as nn
import torch


class SharedEmbedderModel(nn.Module):

    def __init__(self, num_classes=8142, hidden_size=768):
        super(SharedEmbedderModel, self).__init__()

        self.alpha = 1.0
        self.inference_alpha = 0.5

        # shared embedder
        self.embedder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.embedder.head = torch.nn.Identity()

        # classifier branches
        self.cb = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).head
        self.rb = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).head

    def forward(self, img1, img2):
        if img2 is not None:
            # embed
            hidden1 = self.embedder(img1)
            hidden2 = self.embedder(img2)

            # predict
            logits1 = self.cb(hidden1)
            logits2 = self.rb(hidden2)

            return logits1, logits2
        else:
            # embed
            hidden = self.embedder(img1)

            # predict
            logits1 = self.cb(hidden)
            logits2 = self.rb(hidden)

            # combine
            logits = self.inference_alpha * logits1 + (1 - self.inference_alpha) * logits2

            return logits


# TODO: class SeparateEmbedderModel(nn.Module): (embedder not shared)
