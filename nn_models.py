import torch
import torch.nn as nn
import torch.nn.functional as F
import globals

# ----------------------------
# Neural Networks model architectures
# ----------------------------


class Encoder(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(globals.N_INPUTS, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, z_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


# 1st ARCHITECTURE: upsample convolutions to reconstruct the full spectrum from a low-dimensional latent space calculated from the 9 inputs
class SpectralDecoder1(nn.Module):
    def __init__(self, z_dim=128, channels=16, initial_length=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, channels * initial_length)
        self.initial_length = initial_length
        self.channels = channels

        # processing at low-res (16ch x 32len)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )

        # progressively upsample the sequence length
        self.upsample_pipeline = nn.Sequential(
            nn.ConvTranspose1d(32, 24, kernel_size=6, stride=4, padding=1),
            nn.GroupNorm(8, 24),
            nn.SiLU(),
            nn.ConvTranspose1d(24, 16, kernel_size=6, stride=4, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.ConvTranspose1d(16, 12, kernel_size=6, stride=4, padding=1),
            nn.GroupNorm(4, 12),
            nn.SiLU(),
            nn.ConvTranspose1d(12, 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
        )

        # final adjustment to hit exactly 4205 and map to 1 output channel
        self.final_conv = nn.Conv1d(8, 1, kernel_size=5, padding=2)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.channels, self.initial_length)
        x = self.initial_conv(x)
        x = self.upsample_pipeline(x)  # (N, 8, 4096)

        # from 4096, use a tiny interpolation just to cover the last 109 points
        x = F.interpolate(x, size=4205, mode="linear", align_corners=False)

        x = self.final_conv(x).squeeze(1)
        return x
    
class EmulatorSet1(nn.Module):
    def __init__(self, encoder_type="single"):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "single":
            self.encoder = Encoder()
        else:
            self.encoder = nn.ModuleList([
                Encoder() for _ in range(globals.N_FUNCTIONS)
            ])

        # 6 decoders, one for each radiative transfer function
        self.decoders = nn.ModuleList([
            SpectralDecoder1() for _ in range(globals.N_FUNCTIONS)
        ])

    def forward(self, x):
        if self.encoder_type == "single":
            z = self.encoder(x)
            outputs = [
                decoder(z)
                for decoder in self.decoders
            ]
        else:
            outputs = [
                decoder(encoder(x))
                for encoder, decoder in zip(self.encoder, self.decoders)
            ]
            
        # return prediction as one tensor of shape (N, 6, 4205)
        return torch.stack(outputs, dim=1)
    

# 2nd ARCHITECTURE: directly predict the PCA coefficients for each function from the latent space with MLP, without upsampling convolutions
class SpectralDecoder2(nn.Module):
    def __init__(self, z_dim=128, n_components=10):
        super().__init__()
        # map the latent space to the PCA coefficients
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, n_components) # output shape: (N, 10)
        )

    def forward(self, z):
        return self.net(z)

class EmulatorSet2(nn.Module):
    def __init__(self, encoder_type="single", n_components=10):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "single":
            self.encoder = Encoder()
        else:
            self.encoder = nn.ModuleList([
                Encoder() for _ in range(globals.N_FUNCTIONS)
            ])
        
        self.decoders = nn.ModuleList([
            SpectralDecoder2(z_dim=128, n_components=n_components) for _ in range(globals.N_FUNCTIONS)
        ])

    def forward(self, x):
        if self.encoder_type == "single":
            z = self.encoder(x)
            outputs = [
                decoder(z)
                for decoder in self.decoders
            ]
        else:
            outputs = [
                decoder(encoder(x))
                for encoder, decoder in zip(self.encoder, self.decoders)
            ]
        # return prediction as one tensor of shape (N, 6, 10) - PCA coefficients for each function
        return torch.stack(outputs, dim=1)
    

# 3rd ARCHITECTURE: add convolutional processing to the latent space before predicting PCA coefficients, to allow the model to learn local relationships in the structured sequence space
# technically wrong, as the PCA-reduced space doesn't have a true spatial structure, but it allows us to experiment with convolutional processing
class SpectralDecoder3(nn.Module):
    def __init__(self, z_dim=128, n_components=10, initial_length=32, channels=16):
        super().__init__()
        self.initial_length = initial_length
        self.channels = channels

        # map latent space to a structured sequence space
        self.fc = nn.Linear(z_dim, channels * initial_length)

        # convolutions process features across the sequence dimension
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 24, kernel_size=5, padding=2),
            nn.GroupNorm(8, 24),
            nn.SiLU(),
            nn.Conv1d(24, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )

        # collapse the remaining sequence length down to exactly 10 PCA dimensions
        self.to_pca = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * initial_length, 64),
            nn.SiLU(),
            nn.Linear(64, n_components) # final output shape: (N, 10)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.channels, self.initial_length)
        x = self.cnn(x)
        pca_coefficients = self.to_pca(x)
        return pca_coefficients
    
class EmulatorSet3(nn.Module):
    def __init__(self, encoder_type="single", n_components=10):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "single":
            self.encoder = Encoder()
        else:
            self.encoder = nn.ModuleList([
                Encoder() for _ in range(globals.N_FUNCTIONS)
            ])
        
        self.decoders = nn.ModuleList([
            SpectralDecoder3(z_dim=128, n_components=n_components) 
            for _ in range(globals.N_FUNCTIONS)
        ])

    def forward(self, x):
        if self.encoder_type == "single":
            z = self.encoder(x)
            outputs = [
                decoder(z)
                for decoder in self.decoders
            ]
        else:
            outputs = [
                decoder(encoder(x))
                for encoder, decoder in zip(self.encoder, self.decoders)
            ]
        # return prediction as one tensor of shape (N, 6, 10) - PCA coefficients for each function
        return torch.stack(outputs, dim=1)
    

# 4th ARCHITECTURE: use squeeze and excite blocks to allow the model to learn which latent features are most important for each function, before predicting PCA coefficients
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(x)
        return x * weights
    
class SpectralDecoder4(nn.Module):
    def __init__(self, z_dim=128, n_components=10):
        super().__init__()

        self.se = SEBlock(z_dim)

        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, n_components)
        )

    def forward(self, z):
        z = self.se(z)
        return self.mlp(z)
    
class EmulatorSet4(nn.Module):
    def __init__(self, encoder_type="single", n_components=10):
        super().__init__()

        self.encoder_type = encoder_type
        if encoder_type == "single":
            self.encoder = Encoder()
        else:
            self.encoder = nn.ModuleList([
                Encoder() for _ in range(globals.N_FUNCTIONS)
            ])

        self.decoders = nn.ModuleList([
            SpectralDecoder4(z_dim=128, n_components=n_components)
            for _ in range(globals.N_FUNCTIONS)
        ])

    def forward(self, x):
        if self.encoder_type == "single":
            z = self.encoder(x)
            outputs = [
                decoder(z)
                for decoder in self.decoders
            ]
        else:
            outputs = [
                decoder(encoder(x))
                for encoder, decoder in zip(self.encoder, self.decoders)
            ]

        return torch.stack(outputs, dim=1)


# 5th ARCHITECTURE: physics-informed full-spectrum emulator.
# Encoder: symbolic physics features plus learnable multi-output feature selection.
# Decoder: function-correlation mixing plus spectral-region-specific convolutions.
class ResidualFCBlock(nn.Module):
    """A single Linear layer wrapped with a (projected) skip connection.
    Learns a residual on top of an identity/linear-projected input instead
    of having to reconstruct the full mapping from scratch."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )
 
    def forward(self, x):
        return self.act(self.fc(x) + self.skip(x))
 
 
class ResidualConv1D(nn.Module):
    """A single (transpose) conv layer wrapped with a skip connection.
    The skip path 1x1-projects the channels (if needed) and linearly
    resamples to the exact output length of the main path, so it works
    uniformly whether the block upsamples or keeps length fixed."""
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, num_groups, transpose=True):
        super().__init__()
        conv_cls = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.conv = conv_cls(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()
        self.skip_proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
 
    def forward(self, x):
        main = self.conv(x)
        skip = self.skip_proj(x)
        if skip.shape[-1] != main.shape[-1]:
            skip = F.interpolate(skip, size=main.shape[-1], mode="linear", align_corners=False)
        return self.act(self.norm(main + skip))
 
 
class FunctionMixer(nn.Module):
    """Lets the N_FUNCTIONS decoder branches exchange information at a single
    point in the network. Operates purely along the function axis (channel
    and length dimensions are untouched elementwise), so it adds negligible
    parameters (n_functions^2) and is initialized as an exact identity
    mapping (delta = 0), meaning training starts out identical to fully
    independent decoders and only begins mixing once it's useful."""
    def __init__(self, n_functions):
        super().__init__()
        self.n_functions = n_functions
        self.delta = nn.Parameter(torch.zeros(n_functions, n_functions))
 
    def forward(self, feats_list):
        x = torch.stack(feats_list, dim=1)
        f = x.shape[1]
        mix = torch.eye(f, device=x.device, dtype=x.dtype) + self.delta
        x = torch.einsum('fg,bgcl->bfcl', mix, x)
        return [x[:, i] for i in range(f)]
 
 
class GlobalSpectralResidual(nn.Module):
    """Adds a low-rank *global* correction to the local convolutional
    prediction, to capture whole-spectrum correlations that local conv
    kernels can't see (a kernel of size 5 only ever looks at 5 neighboring
    wavelengths, no matter how deep the network is).
 
    Rather than a full (spectrum_len x spectrum_len) correlation/covariance
    matrix (4205^2 ~ 17.7M entries -- far too many parameters), the
    correction is expressed in a small basis of `n_modes` learned global
    spectral shapes: a linear head turns the latent z into n_modes
    coefficients, and the correction is their linear combination with the
    basis. This is the same idea as PCA / empirical-orthogonal-function
    (EOF) decomposition of a spectral covariance matrix -- a handful of
    leading eigenvectors usually capture almost all the shape variation in
    a smooth physical spectrum.
 
    The basis is learned end-to-end by default (random init). Optionally,
    call init_basis_from_data with a batch of real target spectra to seed
    it with the actual leading singular vectors of the data (mathematically
    the eigenvectors of the covariance matrix, computed via SVD so the
    giant spectrum_len x spectrum_len covariance matrix is never formed).
 
    `gate` starts at 0, so this path contributes nothing until training
    finds it useful -- same safety pattern as the residual blocks and the
    FunctionMixer.
    """
    def __init__(self, z_dim, spectrum_len, n_modes=8):
        super().__init__()
        self.n_modes = n_modes
        self.coeff_head = nn.Linear(z_dim, n_modes)
        self.basis = nn.Parameter(torch.randn(n_modes, spectrum_len) * 0.01)
        self.gate = nn.Parameter(torch.zeros(1))
 
    def forward(self, z):
        coeffs = self.coeff_head(z)
        correction = coeffs @ self.basis
        return self.gate * correction
 
    @torch.no_grad()
    def init_basis_from_data(self, spectra):
        """spectra: (N_samples, spectrum_len) tensor of real target spectra
        for this function, e.g. a large batch from the training set.
        Seeds self.basis with the leading right-singular vectors of the
        centered data -- equivalent to the leading eigenvectors of the
        (spectrum_len x spectrum_len) covariance matrix, but computed
        without ever forming that matrix."""
        mean = spectra.mean(dim=0, keepdim=True)
        centered = spectra - mean
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        self.basis.copy_(Vt[: self.n_modes].to(self.basis.dtype))
 
 
class RegionBranchingBlock(nn.Module):
    """Replaces a single global 'upsample + read out' stage with parallel,
    region-specific branches. Each branch covers one contiguous portion of
    the output spectrum and uses its own kernel_size (stride is kept at 2
    everywhere, matching the rest of the upsampling cascade -- only kernel
    size, and the padding needed to hit the right output length, varies per
    region). Wider kernels suit flat/smooth portions of a function; narrow
    kernels suit portions with dense, sharp absorption lines.
 
    regions: list of (start_frac, end_frac, kernel_size) tuples, fractions
    in [0, 1] along the spectrum, covering [0, 1] with no gaps/overlaps.
 
    Output lengths are computed from the standard ConvTranspose1d formula
    and then snapped exactly via linear interpolation (same trick used in
    ResidualConv1D), so rounding a fractional boundary to an integer index
    never leaves the concatenated output a sample short or long.
    """
 
    def __init__(self, in_ch, mid_ch, in_len, out_len, regions, num_groups):
        super().__init__()
        self.out_len = out_len
        self.regions = regions
        self.up_branches = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.readout_branches = nn.ModuleList()
        stride = 2
 
        for (f0, f1, k) in regions:
            i0, i1 = round(f0 * in_len), round(f1 * in_len)
            o0, o1 = round(f0 * out_len), round(f1 * out_len)
            in_span, out_span = i1 - i0, o1 - o0
            pad = max(0, round(((in_span - 1) * stride + k - out_span) / 2))
            self.up_branches.append(
                nn.ConvTranspose1d(in_ch, mid_ch, kernel_size=k, stride=stride, padding=pad)
            )
            self.norms.append(nn.GroupNorm(num_groups, mid_ch))
            self.readout_branches.append(nn.Conv1d(mid_ch, 1, kernel_size=k, padding=k // 2))
 
        self.act = nn.SiLU()
 
    def forward(self, x):
        in_len = x.shape[-1]
        mid_segs, bounds = [], []
        for (f0, f1, k), up, norm in zip(self.regions, self.up_branches, self.norms):
            i0, i1 = round(f0 * in_len), round(f1 * in_len)
            o0, o1 = round(f0 * self.out_len), round(f1 * self.out_len)
            seg = x[:, :, i0:i1]
            seg_up = self.act(norm(up(seg)))
            target_len = o1 - o0
            if seg_up.shape[-1] != target_len:
                seg_up = F.interpolate(seg_up, size=target_len, mode="linear", align_corners=False)
            mid_segs.append(seg_up)
            bounds.append((o0, o1))
        mid = torch.cat(mid_segs, dim=-1)
 
        out_segs = []
        for readout, (o0, o1) in zip(self.readout_branches, bounds):
            out_segs.append(readout(mid[:, :, o0:o1]))
        out = torch.cat(out_segs, dim=-1)
        return mid, out
 
 
class Encoder5(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.block1 = ResidualFCBlock(9, 64)
        self.block2 = ResidualFCBlock(64, 128)
        self.block3 = ResidualFCBlock(128, z_dim)
 
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
 
 
class SpectralDecoder5(nn.Module):
    def __init__(self, z_dim=128, channels=64, initial_length=32, spectrum_len=4205,
                 n_global_modes=8, region_config=None):
        super().__init__()
        self.fc = nn.Linear(z_dim, channels * initial_length)
        self.initial_length = initial_length
        self.channels = channels
 
        self.initial_conv = ResidualConv1D(channels, 128, kernel_size=5, stride=1, padding=2, num_groups=16, transpose=False)
 
        self.up1 = ResidualConv1D(128, 96, kernel_size=5, stride=2, padding=1, num_groups=16, transpose=True)
        self.up2 = ResidualConv1D(96, 64, kernel_size=5, stride=2, padding=1, num_groups=8, transpose=True)
        self.up3 = ResidualConv1D(64, 32, kernel_size=5, stride=2, padding=1, num_groups=8, transpose=True)
        self.up4 = ResidualConv1D(32, 24, kernel_size=5, stride=2, padding=2, num_groups=8, transpose=True)
        self.up5 = ResidualConv1D(24, 16, kernel_size=5, stride=2, padding=1, num_groups=4, transpose=True)
        self.up6 = ResidualConv1D(16, 12, kernel_size=5, stride=2, padding=1, num_groups=4, transpose=True)
 
        # last-mile stage: region-specific (kernel, padding) branches instead
        # of a single global up7 + final_conv. Defaults to one region
        # (kernel=5) if no region_config is given, reproducing the original
        # un-branched behavior exactly.
        self.final_stage = RegionBranchingBlock(
            in_ch=12, mid_ch=8, in_len=2103, out_len=spectrum_len,
            regions=region_config or [(0.0, 1.0, 5)],
            num_groups=4,
        )
 
        self.global_residual = GlobalSpectralResidual(z_dim, spectrum_len, n_modes=n_global_modes)
 
    def decode_stage1(self, z):
        """fc -> initial_conv. Returns the intermediate tensor plus the
        feats collected so far. This is the hand-off point where
        EmulatorSet5 lets the 6 branches exchange information before
        they continue independently."""
        feats = {}
        x = self.fc(z).view(z.size(0), self.channels, self.initial_length)
        feats["32"] = x
        x = self.initial_conv(x)
        feats["32_conv"] = x
        feats["global_correction"] = self.global_residual(z)
        return x, feats
 
    def decode_stage2(self, x, feats):
        """up1 ... final_conv, continuing from decode_stage1's output
        (or from a cross-function-mixed version of it)."""
        x = self.up1(x); feats["65"] = x
        x = self.up2(x); feats["131"] = x
        x = self.up3(x); feats["263"] = x
        x = self.up4(x); feats["525"] = x
        x = self.up5(x); feats["1051"] = x
        x = self.up6(x); feats["2103"] = x
 
        mid, out = self.final_stage(x)
        feats["4205"] = mid          # 8-channel combined feature map (same key as before)
        feats["4205_conv"] = out     # local (conv-only) prediction, pre-global-correction
 
        local_pred = out.squeeze(1)
        combined = local_pred + feats["global_correction"]
        feats["4205_final"] = combined # what's actually returned / trained against
 
        return combined, feats
 
    def forward(self, z, return_features=True):
        """Kept intact for standalone use (e.g. testing/plotting a single
        decoder in isolation, without cross-function mixing)."""
        x, feats = self.decode_stage1(z)
        out, feats = self.decode_stage2(x, feats)
        if return_features:
            return out, feats
        return out
 
 
# Region configs, one list per radiative transfer function, based on visual
# inspection of the provided sample plots. Each tuple is
# (start_frac, end_frac, kernel_size) along the 400-2500nm spectrum.
# IMPORTANT: this assumes decoders[i] corresponds to the i-th function in
# this exact order -- double check this matches your actual dataset/
# training pipeline ordering, and re-tune the fractions once you have the
# real wavelength grid (these are eyeballed from one sample's plot).
REGION_CONFIGS = [
    # 0: Path Radiance -- dense narrow lines 400-900nm, then flat near-zero tail
    [(0.00, 0.24, 3), (0.24, 1.00, 11)],
    # 1: Direct Solar Irradiance -- oscillatory through ~2000nm (incl. the
    # two broad troughs near 1350-1500nm & 1800-1950nm), calmer tail
    [(0.00, 0.26, 3), (0.26, 0.76, 5), (0.76, 1.00, 9)],
    # 2: Diffuse Solar Irradiance -- dense narrow lines 400-1000nm, then flat tail
    [(0.00, 0.29, 3), (0.29, 1.00, 11)],
    # 3: Spherical Albedo -- narrow dips on smooth decay (400-1000nm),
    # broader dips (1000-2000nm), flatter noisy tail (2000-2500nm)
    [(0.00, 0.29, 3), (0.29, 0.76, 5), (0.76, 1.00, 9)],
    # 4: Direct Transmittance -- rising oscillatory + several sharp deep
    # dips through ~2000nm (kept fine to not blur the deep dips), mildly
    # calmer tail
    [(0.00, 0.26, 3), (0.26, 0.76, 3), (0.76, 1.00, 7)],
    # 5: Diffuse Transmittance -- same overall shape family as Spherical Albedo
    [(0.00, 0.29, 3), (0.29, 0.76, 5), (0.76, 1.00, 9)],
]
 
 
class EmulatorSet5(nn.Module):
    def __init__(self, encoder_type="single"):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "single":
            self.encoder = Encoder5()
        else:
            self.encoder = nn.ModuleList([
                Encoder5() for _ in range(globals.N_FUNCTIONS)
            ])
 
        self.decoders = nn.ModuleList([
            SpectralDecoder5(region_config=REGION_CONFIGS[i])
            for i in range(globals.N_FUNCTIONS)
        ])
 
        # single shared module letting the 6 decoder branches exchange
        # information right after initial_conv
        self.function_mixer = FunctionMixer(globals.N_FUNCTIONS)
 
    def forward(self, x):
        if self.encoder_type == "single":
            z = self.encoder(x)
            zs = [z for _ in self.decoders]
        else:
            zs = [encoder(x) for encoder in self.encoder]
 
        stage1 = [decoder.decode_stage1(z) for decoder, z in zip(self.decoders, zs)]
        xs = [s[0] for s in stage1]
        feats_list = [s[1] for s in stage1]
 
        xs = self.function_mixer(xs)
 
        outputs = [
            decoder.decode_stage2(x_i, feats_i)
            for decoder, x_i, feats_i in zip(self.decoders, xs, feats_list)
        ]
 
        model_outputs = []
        model_features = []
        for out in outputs:
            model_outputs.append(out[0])
            model_features.append(out[1])
        return torch.stack(model_outputs, dim=1), model_features