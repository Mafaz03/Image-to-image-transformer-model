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
            **kwargs
    ):
        super().__init__()      
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

def residual_block(x: torch.Tensor, downsample: bool, out_channels: int, kernel_size: int = 3) -> torch.Tensor:
    in_channels = x.shape[1]
    y = nn.Conv2d(
        in_channels=in_channels,  # Specify the correct number of input channels
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=(1 if not downsample else 2),
        padding=kernel_size // 2, 
        bias=True  
    )(x)

    y = relu_bn(y)
    y = nn.Conv2d(
        in_channels=y.shape[1],  
        out_channels=out_channels,  
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2  
    )(y)

    if downsample:
        x = nn.Conv2d(
            in_channels=x.shape[1],  
            out_channels=out_channels,  
            kernel_size=1,
            stride=2,
            padding=0  
        )(x)
    return relu_bn(x+y)

def Generator(image):
    p = Patches(config=TransformerConfig())
    patches = p(image).transpose(1,2)
    pe = PatchEncoder(config=TransformerConfig())
    patch_encoder = pe(patches)

    encoder_layer = TransformerEncoder(TransformerConfig())
    output = encoder_layer(patch_encoder)

    x = output.view(output.shape[0], 1024, 8, 8)
    x = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=6, stride=2, padding=2, bias=False)(x)
    x = nn.BatchNorm2d(num_features=x.shape[1], eps=TransformerConfig().eps)(x)
    x = nn.functional.leaky_relu(x)

    x = residual_block(x, downsample=False, out_channels=512)

    x = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=6, stride=2, padding=2, bias=False)(x)
    x = nn.BatchNorm2d(num_features=x.shape[1], eps=TransformerConfig().eps)(x)
    x = nn.functional.leaky_relu(x)

    x = residual_block(x, downsample=False, out_channels=256)

    x = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=7, stride=2, padding=2, bias=False)(x)
    x = nn.BatchNorm2d(num_features=x.shape[1], eps=TransformerConfig().eps)(x)
    x = nn.functional.leaky_relu(x)

    x = residual_block(x, downsample=False, out_channels=64)

    x = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=7, stride=4, padding=2, bias=False)(x)
    x = nn.BatchNorm2d(num_features=x.shape[1], eps=TransformerConfig().eps)(x)
    x = nn.functional.leaky_relu(x)

    x = residual_block(x, downsample=False, out_channels=32)

    x = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=6, stride=1, padding=1, bias=False)(x)

    x = torch.tanh(x)
    
    return x

if __name__ == "__main__":
    img = torch.rand(1, 3, 256, 256)
    output = Generator(img)
    print(output.shape)
    
