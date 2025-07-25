import torch
import numpy as np

# Simple passthrough encoder: returns input coordinates as-is
class NoEncoding(torch.nn.Module):
    def __init__(self, num_input):
        super().__init__()

        self.encoding_size = num_input # No change to input dimensionality

    def forward(self, coords):
        return coords

# Positional encoding using fixed sine features at exponentially increasing frequencies
class SimpleEncoding(torch.nn.Module):
    def __init__(self,
                 num_input,
                 basis,
                 device):
        super().__init__()

        self.num_input = num_input
        self.basis = basis # Number of frequency bands
        self.device = device

        # Compute 2^i for i in [0, basis)
        self.scales = 2.0 ** torch.arange(0, self.basis).to(self.device)
        # Total encoding size = input + 2*sin/cos per basis per input dim
        self.encoding_size = self.num_input + self.num_input * 2 * self.basis

    def forward(self, coords):
        # coords: [..., D] → xb: [..., basis, D]
        xb = coords[..., None, :] * self.scales[:, None]
        # Compute sin and cos using phase shift trick (sin(x), sin(x + π/2) = cos(x))
        four_feat = torch.sin(torch.stack([xb, xb + 0.5 * torch.pi], axis=-2))
        # Flatten: [..., 2, basis, D] → [..., basis * 2 * D]
        four_feat = four_feat.reshape((*coords.shape[:-1], -1))
        # Concatenate raw coords and sinusoidal features
        values = torch.cat([coords, four_feat], dim=-1)

        return values

# Fourier features with random Gaussian bases
class FourierEncoding(torch.nn.Module):
    def __init__(self,
                 num_input,
                 basis,
                 sigma,
                 device):
        super().__init__()

        self.num_input = num_input
        self.basis = basis

        # Random Gaussian bases scaled by sigma
        self.gaussian = torch.randn([num_input * basis])
        self.coefficients = sigma * self.gaussian
        self.device = device

        # Output is sin and cos per basis per input dim
        self.encoding_size = self.num_input * 2 * self.basis

    def forward(self, coords):
        # Repeat coords to match the number of basis features
        basis_values = torch.cat(self.basis * [coords], dim=-1)
        # Project input through random Fourier bases
        value = 2 * torch.pi * basis_values * self.coefficients
        # Concatenate sin and cos features
        values = torch.cat([torch.sin(value), torch.cos(value)], dim=-1)
        return values

# Learnable, progressively growing frequency encoding (used in FreeNeRF)
# based on https://github.com/Jiawei-Yang/FreeNeRF/blob/main/internal/math.py#L277
class FreeEncoding(torch.nn.Module):
    def __init__(self,
                 num_input,
                 basis,
                 window_start,
                 device):
        super().__init__()

        self.num_input = num_input
        self.basis = basis
        self.window_start = window_start
        self.device = device

        # Total encoded size: original input + 2 features per basis per dim
        self.encoding_size = self.num_input + self.num_input * 2 * self.basis
        # Exponentially increasing frequency scales
        self.scales = 2.0 ** torch.arange(0, basis).to(self.device)
        # Alpha values for masking frequency bands during training
        self.alpha = torch.ones(self.basis).float()
        # Tracks current frequency window
        self.ptr = self.window_start

    def update_alpha(self, current_iter, max_iter):
        # Update the frequency mask (α) progressively with training
        if current_iter < max_iter:
            freq_mask = np.zeros(self.basis)
            # Interpolate the window size from start to full
            ptr = ((self.basis-self.window_start) * current_iter) / max_iter + self.window_start
            self.ptr = ptr
            int_ptr = int(ptr)

            # Fully turn on the lower frequencies
            freq_mask[: int_ptr + 1] = 1.0
            # Linearly interpolate partial activation of the next frequency
            freq_mask[int_ptr : int_ptr + 1] = (ptr - int_ptr)
            # Convert to tensor and clamp for numerical stability
            self.alpha = torch.clip(torch.from_numpy(freq_mask), 1e-8, 1-1e-8).float() # for numerical stability
        else:
            # At end of training, all frequencies are fully active
            self.ptr = self.basis
            self.alpha = torch.ones(self.basis).float()

    def forward(self, coords):
        # coords: [..., D]
        # Multiply input by frequency scales
        xb = coords[..., None, :] * self.scales[:, None]
        # Generate sin and cos features
        four_feat = torch.sin(torch.stack([xb, xb + 0.5 * torch.pi], axis=-2))

        # Apply frequency mask alpha (windowed features)
        window = self.alpha.to(self.device)
        # Broadcast alpha across spatial dims
        four_feat = window[..., None, None] * four_feat

        # Flatten the features: [..., 2, basis, D] → [..., basis * 2 * D]
        four_feat = four_feat.reshape((*coords.shape[:-1], -1))

        # Concatenate input with Fourier features
        fin_values = torch.cat([coords, four_feat], dim=-1)
        return fin_values