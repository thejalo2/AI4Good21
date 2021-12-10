import timm
import torch
import torch.nn as nn
import torch


class SharedEmbedderModel(nn.Module):

    def __init__(self, num_classes=8142, hidden_size=768, share_embedder=True):
        super(SharedEmbedderModel, self).__init__()

        self.alpha = 1.0
        self.inference_alpha = 0.5
        self.share_embedder = share_embedder


        # embedder
        if share_embedder:
            self.embedder = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.embedder.head = torch.nn.Identity()
        else:
            self.embedder_cb = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.embedder_rb = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.embedder_cb.head = torch.nn.Identity()
            self.embedder_rb.head = torch.nn.Identity()

        # classifier branches
        self.cb = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).head
        self.rb = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes).head

    def forward(self, img1, img2, combine=True):
        if img2 is not None:
            # embed
            if self.share_embedder:
                hidden1 = self.embedder(img1)
                hidden2 = self.embedder(img2)
            else:
                hidden1 = self.embedder_cb(img1)
                hidden2 = self.embedder_rb(img2)

            # predict
            logits1 = self.cb(hidden1)
            logits2 = self.rb(hidden2)

            return logits1, logits2
        else:
            # embed
            if self.share_embedder:
                hidden1 = self.embedder(img1)
                hidden2 = hidden1
            else:
                hidden1 = self.embedder_cb(img1)
                hidden2 = self.embedder_rb(img1)

            # predict
            logits1 = self.cb(hidden1)
            logits2 = self.rb(hidden2)

            if combine:
                logits = self.inference_alpha * logits1 + (1 - self.inference_alpha) * logits2
                return logits
            else:
                return logits1, logits2

