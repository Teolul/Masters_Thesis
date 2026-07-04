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
class SymbolicPhysicsFeatures5(nn.Module):
    def __init__(self):
        super().__init__()

        mins = torch.tensor([0.2503, 0.7010, 0.1007, 0.6002, 0.5000, 0.8501, 10.0422, 0.0239, 0.0474])
        maxs = torch.tensor([0.3999, 3.4994, 0.4992, 0.7997, 1.6998, 0.9498, 59.9118, 14.9793, 179.9885])
        means = torch.tensor([0.3250, 2.1000, 0.3000, 0.7000, 1.1000, 0.9000, 35.0001, 7.4998, 90.0019])
        stds = torch.tensor([0.0433, 0.8084, 0.1155, 0.0577, 0.3464, 0.0289, 14.4330, 4.3303, 51.9564])

        self.register_buffer("mins", mins)
        self.register_buffer("ranges", maxs - mins)
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.n_features = 62

    @staticmethod
    def _physics_terms(x, eps=1e-6):
        o3, cwv, aod, asym, ang, ssa, sza, vza, raa = x.unbind(dim=-1)
        sza_rad = torch.deg2rad(sza)
        vza_rad = torch.deg2rad(vza)
        raa_rad = torch.deg2rad(raa)

        cos_sza = torch.cos(sza_rad)
        cos_vza = torch.cos(vza_rad)
        sin_sza = torch.sin(sza_rad)
        sin_vza = torch.sin(vza_rad)
        cos_raa = torch.cos(raa_rad)
        cos_scatter = cos_sza * cos_vza + sin_sza * sin_vza * cos_raa

        return torch.stack([
            torch.exp(-torch.clamp(o3, min=0.0)),
            torch.exp(-torch.clamp(cwv, min=0.0)),
            torch.exp(-torch.clamp(aod, min=0.0)),
            torch.log1p(torch.clamp(aod, min=0.0)),
            aod * ang,
            aod * (1.0 - ssa),
            aod * asym,
            asym * ssa,
            cos_sza,
            cos_vza,
            cos_raa,
            sin_sza,
            sin_vza,
            cos_sza * cos_vza,
            cos_scatter,
            1.0 / torch.clamp(cos_sza, min=eps),
            1.0 / torch.clamp(cos_vza, min=eps),
        ], dim=-1)

    def forward(self, x):
        x_minmax_proxy = self.mins + torch.clamp(x, 0.0, 1.0) * self.ranges
        x_standard_proxy = self.means + x * self.stds
        bias = torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device)

        return torch.cat([
            bias,
            x,
            x * x,
            x_minmax_proxy,
            self._physics_terms(x_minmax_proxy),
            self._physics_terms(x_standard_proxy),
        ], dim=-1)


class PhysicsInformedEncoder5(nn.Module):
    def __init__(self, z_dim=128, n_functions=globals.N_FUNCTIONS):
        super().__init__()
        self.n_functions = n_functions
        self.symbolic = SymbolicPhysicsFeatures5()

        self.shared = nn.Sequential(
            nn.Linear(self.symbolic.n_features, 128),
            nn.SiLU(),
            nn.Linear(128, 192),
            nn.SiLU(),
        )
        self.to_global_z = nn.Sequential(
            nn.Linear(192, z_dim),
            nn.SiLU()
        )

        self.function_feature_logits = nn.Parameter(torch.zeros(n_functions, self.symbolic.n_features))
        self.function_encoder = nn.Sequential(
            nn.Linear(self.symbolic.n_features + 192, 192),
            nn.SiLU(),
            nn.Linear(192, z_dim),
            nn.SiLU()
        )

    def forward(self, x):
        phi = self.symbolic(x)
        shared = self.shared(phi)
        global_z = self.to_global_z(shared)

        gates = torch.sigmoid(self.function_feature_logits).unsqueeze(0)
        selected_phi = phi.unsqueeze(1) * gates
        shared = shared.unsqueeze(1).expand(-1, self.n_functions, -1)
        function_z = self.function_encoder(torch.cat([selected_phi, shared], dim=-1))
        return global_z, function_z


class FunctionCorrelationMixer5(nn.Module):
    def __init__(self, z_dim=128, n_functions=globals.N_FUNCTIONS, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(z_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(z_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 2 * z_dim),
            nn.SiLU(),
            nn.Linear(2 * z_dim, z_dim)
        )
        self.norm2 = nn.LayerNorm(z_dim)
        self.function_embedding = nn.Parameter(torch.zeros(n_functions, z_dim))

    def forward(self, function_z):
        function_z = function_z + self.function_embedding.unsqueeze(0)
        attended, _ = self.attention(function_z, function_z, function_z, need_weights=False)
        function_z = self.norm1(function_z + attended)
        return self.norm2(function_z + self.mlp(function_z))


class RegionalSpectralBlock5(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.regions = [
            (0, 2143, 17),
            (2143, 2895, 11),
            (2895, 3462, 5),
            (3462, 3667, 3),
            (3667, 3858, 7),
            (3858, 3990, 3),
            (3990, 4176, 9),
            (4176, 4205, 3),
        ]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(4, channels),
                nn.SiLU(),
                nn.Conv1d(channels, channels, kernel_size=1),
            )
            for _, _, kernel_size in self.regions
        ])

    def forward(self, x):
        pieces = []
        for (start, end, _), branch in zip(self.regions, self.branches):
            x_region = x[:, :, start:end]
            pieces.append(x_region + branch(x_region))
        return torch.cat(pieces, dim=-1)


class PhysicsInformedSpectralDecoder5(nn.Module):
    def __init__(self, z_dim=128, channels=16, initial_length=32, n_functions=globals.N_FUNCTIONS):
        super().__init__()
        self.channels = channels
        self.initial_length = initial_length
        self.n_functions = n_functions

        self.function_mixer = FunctionCorrelationMixer5(z_dim=z_dim, n_functions=n_functions)
        self.fc = nn.Linear(2 * z_dim, channels * initial_length)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        self.upsample_pipeline = nn.Sequential(
            nn.ConvTranspose1d(32, 24, kernel_size=6, stride=4, padding=1),
            nn.GroupNorm(8, 24),
            nn.SiLU(),
            nn.ConvTranspose1d(24, 16, kernel_size=6, stride=4, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.ConvTranspose1d(16, channels, kernel_size=6, stride=4, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )
        self.regional_refiner = RegionalSpectralBlock5(channels)
        self.to_function = nn.Conv1d(channels, 1, kernel_size=5, padding=2)
        self.output_function_mixer = nn.Conv1d(n_functions, n_functions, kernel_size=1, bias=False)
        self._init_output_function_mixer()

    def _init_output_function_mixer(self):
        corr = torch.tensor([
            [1.0000, 0.4844, 0.9042, 0.9225, -0.2214, 0.8080],
            [0.4844, 1.0000, 0.6361, 0.5397, 0.3282, 0.5110],
            [0.9042, 0.6361, 1.0000, 0.8989, -0.1521, 0.9143],
            [0.9225, 0.5397, 0.8989, 1.0000, -0.1033, 0.9291],
            [-0.2214, 0.3282, -0.1521, -0.1033, 1.0000, -0.0617],
            [0.8080, 0.5110, 0.9143, 0.9291, -0.0617, 1.0000],
        ])
        weight = torch.eye(self.n_functions) + 0.03 * (corr - torch.eye(self.n_functions))
        with torch.no_grad():
            self.output_function_mixer.weight.copy_(weight.unsqueeze(-1))

    def forward(self, global_z, function_z):
        function_z = self.function_mixer(function_z)
        global_z = global_z.unsqueeze(1).expand(-1, self.n_functions, -1)
        z = torch.cat([global_z, function_z], dim=-1)

        x = self.fc(z).view(-1, self.channels, self.initial_length)
        x = self.initial_conv(x)
        x = self.upsample_pipeline(x)
        x = F.interpolate(x, size=4205, mode="linear", align_corners=False)
        x = self.regional_refiner(x)
        x = self.to_function(x).view(-1, self.n_functions, 4205)
        return self.output_function_mixer(x)


class EmulatorSet5(nn.Module):
    def __init__(self, encoder_type="single"):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "single":
            self.encoder = PhysicsInformedEncoder5()
        else:
            self.encoder = nn.ModuleList([
                PhysicsInformedEncoder5(n_functions=1)
                for _ in range(globals.N_FUNCTIONS)
            ])

        self.decoder = PhysicsInformedSpectralDecoder5()

    def forward(self, x):
        if self.encoder_type == "single":
            global_z, function_z = self.encoder(x)
        else:
            global_parts = []
            function_parts = []
            for encoder in self.encoder:
                global_z_i, function_z_i = encoder(x)
                global_parts.append(global_z_i)
                function_parts.append(function_z_i)

            global_z = torch.stack(global_parts, dim=1).mean(dim=1)
            function_z = torch.cat(function_parts, dim=1)

        return self.decoder(global_z, function_z)
