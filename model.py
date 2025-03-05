import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

class TransformerConfig:
    def __init__(
            self,
            patch_size = 8,
            projection_dim = 64,
            image_size = 256,
            eps = 1e-6,
            channels = 3,
            num_heads = 2,
            dropout = 0.5,
            intermediate_size = 128,
            num_hidden_layers=12,
            epochs = 10,
            lr = 2e-4,
            train_bs = 7,
            test_bs = 2,
            **kwargs
    ):
        super().__init__()    
        self.epochs = epochs  
        self.lr = lr
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.image_size = image_size
        self.eps = eps
        self.channels = channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        

class Patches:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.patch_size = config.patch_size
    def __call__(self, images):
        # Input: [Batch_Size, Channels, Height, Width]
        # Output: [Batch_Size, Patch_Dim, Num_Patches]
        # Num_Patches = (Height//Patch_Size) * (Width//Patch_Size)
        # Patch_Dims = Channels * Patch_Size * Patch_Size
        assert (images.shape[2] >= self.patch_size) and (images.shape[3] >= self.patch_size), f"Image size: {images.shape} must be bigger or equal to patch size: {self.patch_size}"
        patches = F.unfold(images, 
                           kernel_size=self.patch_size, 
                           stride=self.patch_size)
        return patches
    
class PatchEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_patches = (config.image_size//config.patch_size)**2
        self.projection = nn.Linear(config.patch_size * config.patch_size * config.channels, config.projection_dim)
        self.position_embedding = nn.Embedding(self.num_patches, config.projection_dim)
    def forward(self, patch):
        positions = torch.arange(start=0, end = self.num_patches, device= patch.device)
        a = self.projection(patch)
        b = self.position_embedding(positions)
        encoded = a + b
        return encoded

class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed_dim = config.projection_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # 1/root(head_dim)
        self.dropout = config.dropout

        self.k_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)
        self.v_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)
        self.q_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)

        self.out_proj = nn.Linear(in_features = self.embed_dim, out_features = self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch. Tensor, torch. Tensor]:
        batch_size, seq_len, embed_dim = hidden_states.size() # seq_len is same as Num_Patches and embed_dim is same as projection or hdden size

        query_states = self.q_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)   # [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states) # [Batch_Size, Num_Patches, Embed_Dim]

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)     # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]

        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] * [Batch_Size, Num_Heads, Head_Dim, Num_Patches] -> [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        # [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        #[Batch_Size, Num_Heads, Num_Patches, Num_Patches] * [Batch_Size, Num_Heads, Num_Patches, Head_Dim] = 
        attn_output = torch.matmul(attn_weights, value_states) # [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)} but is {attn_weights.size()}")
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim) # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output) # [Batch_Size, Num_Patches, Embed_Dim]

        return attn_output, attn_weights

class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layer_1 = nn.Linear(config.projection_dim, config.intermediate_size)
        self.layer_2 = nn.Linear(config.intermediate_size, config.projection_dim)
    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.layer_1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        return self.layer_2(hidden_states)

class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embeds = config.projection_dim
        self.attn = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(self.embeds, eps=config.eps)
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(self.embeds, eps=config.eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm_1(hidden_states)
        hidden_states, _ = self.attn(hidden_states)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.layernorm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    def forward(self, hidden_states):
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states

def relu_bn(inputs: torch.Tensor) -> torch.Tensor:
    relu = nn.functional.relu(inputs)
    bn = nn.BatchNorm2d(num_features=inputs.shape[1], eps=TransformerConfig().eps)(relu)
    return bn

import torch
import torch.nn as nn

def relu_bn(x):
    device = x.device  # Get the device of the input tensor
    return nn.ReLU()(nn.BatchNorm2d(x.size(1)).to(device)(x))  # Move BatchNorm2d to the same device as x
class ResidualBlock(nn.Module):
    def __init__(self, downsample: bool, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        
        # First convolution layer
        self.y_1 = nn.Conv2d(
            in_channels=in_channels,  
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1 if not downsample else 2),
            padding=kernel_size // 2,
            bias=False
        )
        
        # Second convolution layer
        self.y_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )

        # Downsample shortcut if needed
        if downsample:
            self.downsample_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False
            )
        else:
            self.downsample_conv = None

    def forward(self, x: torch.Tensor):
        # Apply the first convolution and batch normalization + ReLU
        y = relu_bn(self.y_1(x))

        # Apply the second convolution
        y = self.y_2(y)

        # If downsampling, apply the downsample shortcut
        if self.downsample:
            x = self.downsample_conv(x)

        # Add the input (x) to the output (y) and apply ReLU
        return relu_bn(x + y)

# def Generator(image):

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.p = Patches(config=TransformerConfig())  # Assumes you have a Patches class
        self.pe = PatchEncoder(config=TransformerConfig())  # Assumes you have a PatchEncoder class
        self.encoder_layer = TransformerEncoder(TransformerConfig())  # Assumes you have TransformerEncoder class

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(512, eps=TransformerConfig().eps)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(256, eps=TransformerConfig().eps)

        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=7, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64, eps=TransformerConfig().eps)

        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=7, stride=4, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(32, eps=TransformerConfig().eps)

        self.final_conv = nn.Conv2d(32, 3, kernel_size=6, stride=1, padding=1, bias=False)

    def forward(self, image):
        patches = self.p(image).transpose(1, 2)  # Divide image into patches and transpose
        patch_encoder = self.pe(patches)  # Encode patches
        output = self.encoder_layer(patch_encoder)  # Pass through the transformer encoder

        x = output.view(output.shape[0], 1024, 8, 8)  # Reshape the output to the correct spatial dimensions

        x = self.deconv1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)  # LeakyReLU with slope 0.2

        # Using residual_block as a class, initialize with required parameters
        res_block = ResidualBlock(downsample=False, in_channels=512, out_channels=512).to(x.device)
        x = res_block(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)

        res_block = ResidualBlock(downsample=False, in_channels=256, out_channels=256).to(x.device)
        x = res_block(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)

        res_block = ResidualBlock(downsample=False, in_channels=64, out_channels=64).to(x.device)
        x = res_block(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)

        res_block = ResidualBlock(downsample=False, in_channels=32, out_channels=32).to(x.device)
        x = res_block(x)

        x = self.final_conv(x)
        x = torch.tanh(x)  # Use tanh activation to output values between -1 and 1 (normalized RGB)

        return x

if __name__ == "__main__":
    img = torch.rand(1, 3, 256, 256)
    gen = Generator()
    output = gen(img)
    print(output.shape)
    
