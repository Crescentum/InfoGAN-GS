import torch
import torch.nn as nn

NOISE_DIM   = 62   
CAT_DIM     = 10   
CONT_DIM    = 2   
LATENT_DIM  = NOISE_DIM + CAT_DIM + CONT_DIM   
Q_OUT_DIM = CAT_DIM + CONT_DIM * 2   

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)

class Generator(nn.Module):

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        out = self.fc(z)                        
        out = out.view(-1, 128, 7, 7)          
        img = self.deconv(out)                
        return img


class DiscriminatorQ(nn.Module):

    def __init__(self, q_out_dim: int = Q_OUT_DIM):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1024, bias=False),
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
        feat = self.shared_conv(x)     
        feat = self.shared_fc(feat)   

        d_out = self.d_head(feat)     
        q_out = self.q_head(feat)       

        return d_out, q_out


def parse_q_output(q_out: torch.Tensor):
    cat_logits = q_out[:, :CAT_DIM]              
    cont_mean  = q_out[:, CAT_DIM: CAT_DIM + CONT_DIM]         
    cont_logstd = q_out[:, CAT_DIM + CONT_DIM:]               

    cat_prob  = torch.softmax(cat_logits, dim=1)
    cont_std  = torch.exp(cont_logstd)            

    return cat_prob, cont_mean, cont_std


def sample_latent(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    z_noise = torch.FloatTensor(batch_size, NOISE_DIM).uniform_(-1, 1).to(device)

    cat_idx = torch.randint(0, CAT_DIM, (batch_size,), device=device)
    c_cat   = torch.zeros(batch_size, CAT_DIM, device=device)
    c_cat.scatter_(1, cat_idx.unsqueeze(1), 1.0)

    c_cont  = torch.FloatTensor(batch_size, CONT_DIM).uniform_(-1, 1).to(device)

    return z_noise, c_cat, c_cont


def concat_latent(z_noise: torch.Tensor,
                  c_cat:   torch.Tensor,
                  c_cont:  torch.Tensor) -> torch.Tensor:
    return torch.cat([z_noise, c_cat, c_cont], dim=1)
