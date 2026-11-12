import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C, DotProduct

import globals

kern_rbf = (
        C(1.0, (1e-3, 1e3)) *
        RBF(length_scale=np.ones(globals.N_INPUTS), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
    )

kern_matern = (
    C(1.0, (1e-3, 1e3))
    * Matern(
        length_scale=np.ones(globals.N_INPUTS),
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5
    )
    + WhiteKernel(1e-2, (1e-5, 1e1))
)

kern_rq = (
    C(1.0, (1e-3, 1e3))
    * RationalQuadratic(
        length_scale=1.0,
        alpha=1.0,
        length_scale_bounds=(1e-3, 1e3),
        alpha_bounds=(1e-3, 1e3)
    )
    + WhiteKernel(1e-2, (1e-5, 1e1))
)

kern_rbf_rq = (
    C(1.0, (1e-3, 1e3))
    * (
        RBF(
            length_scale=np.ones(globals.N_INPUTS),
            length_scale_bounds=(1e-3, 1e3)
        )
        + RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-3, 1e3),
            alpha_bounds=(1e-3, 1e3)
        )
    )
    + WhiteKernel(1e-2, (1e-5, 1e1))
)

kern_dot_rbf = (
    C(1.0, (1e-3, 1e3))
    * (
        DotProduct()
        + RBF(
            length_scale=np.ones(globals.N_INPUTS),
            length_scale_bounds=(1e-3, 1e3)
        )
    )
    + WhiteKernel(1e-2, (1e-5, 1e1))
)

kern_linear = (
    C(1.0, (1e-3, 1e3))
    * DotProduct()
    + WhiteKernel(1e-2, (1e-5, 1e1))
)

kern_matern_rq = (
    C(1.0, (1e-3, 1e3))
    * (
        Matern(
            length_scale=np.ones(globals.N_INPUTS),
            nu=2.5
        )
        + RationalQuadratic(
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-3, 1e3),
            alpha_bounds=(1e-3, 1e3)
        )
    )
    + WhiteKernel(1e-2, (1e-5, 1e1))
)

kern_matern_iso = (
    C(1.0, (1e-3, 1e3))
    * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5
    )
    + WhiteKernel(1e-2, (1e-5, 1e1))
)