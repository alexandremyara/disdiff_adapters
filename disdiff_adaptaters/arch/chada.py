from lightning import LightningModule
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from disdiff_adaptaters.utils.const import ChAda
from disdiff_adaptaters.utils.utils import collate_images

from chada.backbones.vit.chada_vit import ChAdaViT

class ChAdaViTModule(LightningModule) :
    """
    Lightning interface to use ChAdaVit as an encoder.
    """

    def __init__(self, 
                 patch_size: int, 
                 embed_dim: int,
                 return_all_tokens: bool,
                 max_number_channels: int,
                 mixed_channels=True) :
        
        super().__init__()
        self.save_hyperparameters()

        self.model = ChAdaViT(patch_size=self.hparams.patch_size,
                              embed_dim=self.hparams.embed_dim,
                              return_all_tokens=self.hparams.return_all_tokens,
                              max_number_channels=self.max_number_channels)
        self.targets = []
        self.embeddings=[]

    def forward(self, x) :
        return self.model(x)
    
    def training_step(self, batch):
        #We use only the model.eval mode
        pass
    def validation_step(self, batch):
        #We use only the model.eval mode
        pass

    def test_step(self, batch) :

        #Set the torch weights
        state = torch.load(ChAda.Path.WEIGHTS, map_location="cpu", weights_only=False)["state_dict"]
        for k in list(state.keys()):
            if "encoder" in k:
                state[k.replace("encoder", "backbone")] = state[k]
            if "backbone" in k:
                state[k.replace("backbone.", "")] = state[k]
            del state[k]
        self.model.load_state_dict(state, strict=False)

        #unpack batch
        images, labels = batch

        #Format batch in order to collate
        data = []
        for image, label in zip(images, labels) :
            data.append((image, label))
        #Collate images
        channels_list, labels_list, num_channels_list = collate_images(data)

        #Evaluation
        feats = self.model(channels_list, index=0, list_num_channel=num_channels_list)

        self.embeddings.append(feats)
        self.targets.append(labels)

    def on_test_end(self) :
        z_proj = PCA(n_components=2).fit_transform(self.embeddings)
        shape_labels = self.targets[:, 1]  # par exemple, 1 = shape factor

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z_proj[:, 0], z_proj[:, 1], c=shape_labels, cmap='tab10', s=5)
        plt.colorbar(scatter, label="Shape")
        plt.title("t-SNE des embeddings - coloré par shape")
        plt.show()
        pass
        