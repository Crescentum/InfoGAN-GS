import torch
import torch.nn as nn
import torch.nn.functional as F


NOISE_DIM   = 128
CAT_DIMS    = (20, 20, 20)  
CAT_DIM     = sum(CAT_DIMS) 
N_CATS      = 3
CONT_DIM    = 1             
LATENT_DIM  = NOISE_DIM + CAT_DIM + CONT_DIM   # 189

Q_OUT_DIM = CAT_DIM + CONT_DIM * 2   


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator(nn.Module):

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 8 * 8 * 256),
            nn.BatchNorm1d(8 * 8 * 256),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            # (256, 8, 8) → (256, 8, 8)  [kernel=3, stride=1 keeps spatial size]
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 8, 8) → (256, 8, 8)
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 32, 32) → (1, 64, 64)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)                        # (B, 8*8*256)
        out = out.view(-1, 256, 8, 8)           # (B, 256, 8, 8)
        img = self.deconv(out)                  # (B, 1, 64, 64)
        return img

class DiscriminatorQ(nn.Module):


    def __init__(self, q_out_dim: int = Q_OUT_DIM):
        super().__init__()

        self.shared_conv = nn.Sequential(
            # (1, 64, 64) → (64, 32, 32)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            # (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # (256, 8, 8) → (256, 8, 8)  [kernel=3, stride=1 keeps size]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # (256, 8, 8) → (256, 8, 8)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.d_head = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, q_out_dim),
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor):
        feat = self.shared_conv(x)      # (B, 256, 8, 8)
        feat = self.shared_fc(feat)     # (B, 1024)
        d_out = self.d_head(feat)       # (B, 1)
        q_out = self.q_head(feat)       # (B, 62)
        return d_out, q_out


def parse_q_output(q_out: torch.Tensor):

    cat_logits = []
    offset = 0
    for dim in CAT_DIMS:
        cat_logits.append(q_out[:, offset:offset + dim])
        offset += dim

    cat_prob = [torch.softmax(logits, dim=1) for logits in cat_logits]
    cont_mean = q_out[:, offset:offset + CONT_DIM]
    cont_logstd = q_out[:, offset + CONT_DIM:offset + CONT_DIM * 2]
    cont_std = torch.exp(cont_logstd)
    return cat_prob, cont_mean, cont_std


def sample_latent(batch_size: int, device: torch.device, temperature: float = 1.0):

    z_noise = torch.FloatTensor(batch_size, NOISE_DIM).uniform_(-1, 1).to(device)

    if temperature < 0.0:
        c_cat = torch.zeros(batch_size, CAT_DIM, device=device)
        offset = 0
        for dim in CAT_DIMS:
            cat_idx = torch.randint(0, dim, (batch_size,), device=device)
            c_cat.scatter_(1, (cat_idx + offset).unsqueeze(1), 1.0)
            offset += dim

        c_cont = torch.FloatTensor(batch_size, CONT_DIM).uniform_(-1, 1).to(device)

        return z_noise, c_cat, c_cont
    
    else:
        cat_blocks = []
        for dim in CAT_DIMS:
            logits = torch.zeros(batch_size, dim, device=device)   # 均匀分布
            one_hot = F.gumbel_softmax(logits, tau=temperature, hard=True)
            cat_blocks.append(one_hot)
        c_cat = torch.cat(cat_blocks, dim=1)   # shape (B, sum(CAT_DIMS))

        c_cont = torch.FloatTensor(batch_size, CONT_DIM).uniform_(-1, 1).to(device)

        return z_noise, c_cat, c_cont


def concat_latent(z_noise: torch.Tensor,
                  c_cat: torch.Tensor,
                  c_cont: torch.Tensor) -> torch.Tensor:
    return torch.cat([z_noise, c_cat, c_cont], dim=1)

