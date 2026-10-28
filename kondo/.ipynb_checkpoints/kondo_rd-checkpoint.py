"""
kondo_rd.py
===========
Three-component reaction–diffusion model from

    Kondo S. (2009) Interactions between zebrafish pigment cells
    responsible for the generation of Turing patterns.
    PNAS 106, 8429–8434.  doi:10.1073/pnas.0808622106

PDEs on a 2-D grid with Neumann (zero-flux) boundary conditions:

    ∂u/∂t = F(u,v,w) − c_u·u + D_u·∇²u    [melanophores]
    ∂v/∂t = G(u,v,w) − c_v·v + D_v·∇²v    [xanthophores]
    ∂w/∂t = H(u,v,w) − c_w·w + D_w·∇²w    [long-range factor from u]

Clamped-linear production functions:
    F = clip( c₁v + c₂w + c₃ ,  0, U )
    G = clip( c₄u + c₅w + c₆ ,  0, V )
    H = clip( c₇u + c₈v + c₉ ,  0, W )
"""

import warnings
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =============================================================================
# 1.  Parameter dictionaries
# =============================================================================

WILDTYPE = dict(
    name="Wild Type",
    # production coefficients ──────────────────────────────────
    c1=-0.040,  c2=-0.055,  c3=0.370,   # F: u production
    c4=-0.050,  c5= 0.000,  c6=0.250,   # G: v production
    c7= 0.016,  c8=-0.030,  c9=0.240,   # H: w production
    # degradation rates ────────────────────────────────────────
    cu=0.020,   cv=0.025,   cw=0.060,
    # diffusion coefficients ───────────────────────────────────
    Du=0.020,   Dv=0.020,   Dw=0.200,
    # production upper bounds ──────────────────────────────────
    U=0.5,      V=0.5,      W=0.5,
)

# Leopard mutant: only c3 is raised
LEOPARD = {**WILDTYPE, "name": "Leopard Mutant",           "c3": 0.385}

# Jaguar/Obelix mutant: all diffusion coefficients doubled
JAGUAR  = {**WILDTYPE, "name": "Jaguar/Obelix Mutant",
           "Du": 0.04, "Dv": 0.04, "Dw": 0.40}


# =============================================================================
# 2.  Clamped production functions  (vectorised)
# =============================================================================

def F_prod(v, w, p):
    """u production  clip(c₁v + c₂w + c₃, 0, U)."""
    return np.clip(p["c1"]*v + p["c2"]*w + p["c3"], 0.0, p["U"])

def G_prod(u, w, p):
    """v production  clip(c₄u + c₅w + c₆, 0, V)."""
    return np.clip(p["c4"]*u + p["c5"]*w + p["c6"], 0.0, p["V"])

def H_prod(u, v, p):
    """w production  clip(c₇u + c₈v + c₉, 0, W)."""
    return np.clip(p["c7"]*u + p["c8"]*v + p["c9"], 0.0, p["W"])


# =============================================================================
# 3.  Discrete Laplacian — Neumann (zero-flux) boundary conditions
# =============================================================================

def laplacian_neumann(f):
    """
    5-point Laplacian (dx = dy = 1) with zero-flux BCs.

    Ghost cells are set equal to their nearest interior neighbour
    ('edge' padding), which enforces  ∂f/∂n = 0  on all four walls.
    """
    g = np.pad(f, 1, mode="edge")
    return (g[:-2, 1:-1] + g[2:,  1:-1]   # south + north
          + g[1:-1, :-2] + g[1:-1, 2:]    # west  + east
          - 4.0 * g[1:-1, 1:-1])           # centre


# =============================================================================
# 4.  Reaction Jacobian  (linear production zone)
# =============================================================================

def reaction_jacobian(p):
    """
    J = ∂(rhs) / ∂(u, v, w)  evaluated in the interior of the
    linear production zone (no clipping active):

        ⎡ −cu   c1   c2 ⎤
    J = ⎢  c4  −cv   c5 ⎥
        ⎣  c7   c8  −cw ⎦

    Note: c3, c6, c9, and the diffusion coefficients do NOT appear in J.
    Hence Wild Type, Leopard, and Jaguar share the same J (and therefore
    the same k = 0 stability).  Turing differences arise from nonlinear
    effects (Leopard) or from the shifted dispersion curve (Jaguar).
    """
    return np.array([
        [-p["cu"],  p["c1"],  p["c2"]],
        [ p["c4"], -p["cv"],  p["c5"]],
        [ p["c7"],  p["c8"], -p["cw"]],
    ])


# =============================================================================
# 5.  Steady-state analysis
# =============================================================================

def _ode_residual(x, p):
    """
    Residual vector of the homogeneous (no-diffusion) ODE
    at state x = [u, v, w].
    """
    u, v, w = x[0], x[1], x[2]
    F = np.clip(p["c1"]*v + p["c2"]*w + p["c3"], 0.0, p["U"])
    G = np.clip(p["c4"]*u + p["c5"]*w + p["c6"], 0.0, p["V"])
    H = np.clip(p["c7"]*u + p["c8"]*v + p["c9"], 0.0, p["W"])
    return [F - p["cu"]*u, G - p["cv"]*v, H - p["cw"]*w]


def find_steady_states(p, tol=1e-11):
    """
    Locate all non-negative homogeneous steady states by scanning a
    grid of initial guesses and running scipy.fsolve from each.

    Returns a list of 1-D float arrays  [u*, v*, w*].
    """
    u_max = p["U"] / p["cu"]
    v_max = p["V"] / p["cv"]
    w_max = p["W"] / p["cw"]

    # Analytical linear-range guess:  J · x* = −[c3, c6, c9]
    J = reaction_jacobian(p)
    try:
        x_lin = np.linalg.solve(
            J, -np.array([p["c3"], p["c6"], p["c9"]])).tolist()
    except np.linalg.LinAlgError:
        x_lin = None

    guesses = [
        [u0, v0, w0]
        for u0 in np.linspace(u_max * 0.05, u_max, 7)
        for v0 in np.linspace(v_max * 0.05, v_max, 6)
        for w0 in np.linspace(w_max * 0.05, w_max, 5)
    ] + [
        [1e-3, v_max * 0.95, 1e-3],   # near all-xanthophore boundary state
        [u_max * 0.95, 1e-3, 1e-3],   # near all-melanophore state
        [1e-3, 1e-3, 1e-3],
    ]
    if x_lin is not None:
        guesses.insert(0, x_lin)      # analytical guess first

    found = []
    for x0 in guesses:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sol, _, ier, _ = fsolve(
                    _ode_residual, x0, args=(p,), full_output=True)
            except Exception:
                continue
        if ier != 1:
            continue
        sol = np.asarray(sol, dtype=float)
        if np.any(sol < -1e-8):
            continue
        sol = np.maximum(sol, 0.0)
        if np.max(np.abs(_ode_residual(sol, p))) > tol:
            continue
        if all(np.linalg.norm(sol - s) > 1e-5 for s in found):
            found.append(sol)

    return found


def interior_steady_state(p):
    """
    Return the steady state that lies strictly inside the linear
    production zone of all three functions.  Falls back to the
    analytical linear-range solution if none is classified as interior.
    """
    for s in find_steady_states(p):
        u, v, w = s
        if (0 < p["c1"]*v + p["c2"]*w + p["c3"] < p["U"]
                and 0 < p["c4"]*u + p["c5"]*w + p["c6"] < p["V"]
                and 0 < p["c7"]*u + p["c8"]*v + p["c9"] < p["W"]):
            return s
    # Fallback: solve the linear system directly
    J = reaction_jacobian(p)
    try:
        return np.maximum(
            np.linalg.solve(J, -np.array([p["c3"], p["c6"], p["c9"]])), 0.0)
    except np.linalg.LinAlgError:
        return np.array([p["U"]/(2*p["cu"]),
                         p["V"]/(2*p["cv"]),
                         p["W"]/(2*p["cw"])])


# =============================================================================
# 6.  Dispersion relation and Turing-instability analysis
# =============================================================================

def dispersion_relation(p, k_max=2.5, n_k=2000):
    """
    Compute the dispersion relation

        σ(k)  =  max  Re{ eigenvalues of  J_k }

    where  J_k = J − k²·diag(Du, Dv, Dw).

    Turing instability requires  σ(0) < 0  AND  σ(k*) > 0  for some k*.

    Returns
    -------
    k     : (n_k+1,)  wavenumber array
    sigma : (n_k+1,)  maximum real growth rate at each k
    """
    k_arr  = np.linspace(0.0, k_max, n_k + 1)
    J      = reaction_jacobian(p)
    D_diag = np.diag([p["Du"], p["Dv"], p["Dw"]])
    sigma  = np.array([
        np.max(np.linalg.eigvals(J - ki**2 * D_diag).real)
        for ki in k_arr
    ])
    return k_arr, sigma


def stability_analysis(p, verbose=True):
    """
    Full linear stability analysis around the homogeneous steady state.

    Returns a dict:
      stable, turing, evals0, k_peak, s_peak, lam_dom,
      k_lo, k_hi, k (array), sigma (array).
    """
    J      = reaction_jacobian(p)
    evals0 = np.linalg.eigvals(J)
    stable = bool(np.all(evals0.real < 0))

    k, sigma = dispersion_relation(p)
    pos      = (k > 0) & (sigma > 0)
    turing   = stable and bool(np.any(pos))

    if turing:
        idx_pk  = int(np.argmax(sigma[k > 0]))
        k_peak  = float(k[k > 0][idx_pk])
        s_peak  = float(sigma[k > 0][idx_pk])
        lam_dom = 2.0 * np.pi / k_peak
        k_lo    = float(k[pos][0])
        k_hi    = float(k[pos][-1])
    else:
        k_peak = s_peak = lam_dom = k_lo = k_hi = None

    result = dict(
        name=p["name"], stable=stable, turing=turing,
        evals0=evals0, k_peak=k_peak, s_peak=s_peak,
        lam_dom=lam_dom, k_lo=k_lo, k_hi=k_hi,
        k=k, sigma=sigma,
    )

    if verbose:
        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  {p['name']}")
        print(f"{sep}")
        ev_parts = []
        for e in sorted(evals0, key=lambda x: x.real):
            s = f"{e.real:+.5f}"
            if abs(e.imag) > 1e-9:
                s += f"{e.imag:+.5f}j"
            ev_parts.append(s)
        print(f"  k=0 eigenvalues:        {',  '.join(ev_parts)}")
        print(f"  Homogeneous SS stable:  {'YES ✓' if stable else 'NO  ✗'}")
        print(f"  Turing instability:     {'YES ✓' if turing else 'NO  ✗'}")
        if turing:
            print(f"  Unstable k-band: k ∈ ({k_lo:.4f}, {k_hi:.4f})")
            print(f"  Dominant mode:   k* = {k_peak:.4f}")
            print(f"  Dom. wavelength: λ* = {lam_dom:.2f} grid units")
            print(f"  Peak growth:     σ* = {s_peak:.6f} / time-unit")

    return result


# =============================================================================
# 7.  Simulation  (explicit Euler, Neumann BCs)
# =============================================================================

def simulate(p, n=128, dt=0.5, n_steps=60_000,
             save_every=3_000, seed=42, noise_rel=0.01, verbose=True):
    """
    Forward-Euler simulation of the Kondo (2009) RD system on an
    n×n grid with Neumann (zero-flux) boundary conditions.

    Parameters
    ----------
    p          : dict    parameter set (WILDTYPE / LEOPARD / JAGUAR)
    n          : int     grid size (n × n)
    dt         : float   time step  [must satisfy dt ≤ 1/(4·max(D))]
    n_steps    : int     total number of integration steps
    save_every : int     snapshot interval (in steps)
    seed       : int     NumPy RNG seed for the initial noise
    noise_rel  : float   initial noise amplitude as a fraction of SS value
    verbose    : bool    print progress

    Returns
    -------
    dict with keys:
      'params'    – the parameter dict used
      'ss'        – interior steady state  [u*, v*, w*]
      'snapshots' – list of {'step','time','u','v','w'} dicts
      'final'     – copy of the very last state (same format)
    """
    # ── CFL stability check ──────────────────────────────────────────────────
    D_max    = max(p["Du"], p["Dv"], p["Dw"])
    dt_limit = 1.0 / (4.0 * D_max)
    if dt > dt_limit:
        raise ValueError(
            f"dt={dt:.4f} exceeds the CFL stability limit "
            f"dt_max={dt_limit:.4f}  (D_max={D_max}).\n"
            f"Please use  dt ≤ {dt_limit:.4f}."
        )

    if verbose:
        print(f"\n{'═'*52}")
        print(f"  Simulating: {p['name']}")
        print(f"  Grid {n}×{n}  |  dt={dt}  |  "
              f"steps={n_steps:,d}  |  T_final={n_steps*dt:.0f}")
        print(f"  CFL limit: dt ≤ {dt_limit:.3f}")

    # ── Initial conditions ────────────────────────────────────────────────────
    ss = interior_steady_state(p)
    if verbose:
        resid = max(abs(r) for r in _ode_residual(ss, p))
        print(f"  Interior SS: u*={ss[0]:.4f}, v*={ss[1]:.4f}, "
              f"w*={ss[2]:.4f}   (max|res|={resid:.2e})")

    rng = np.random.default_rng(seed)
    u = np.maximum(ss[0] * (1.0 + noise_rel * rng.standard_normal((n, n))), 0.0)
    v = np.maximum(ss[1] * (1.0 + noise_rel * rng.standard_normal((n, n))), 0.0)
    w = np.maximum(ss[2] * (1.0 + noise_rel * rng.standard_normal((n, n))), 0.0)

    snapshots = [{"step": 0, "time": 0.0,
                  "u": u.copy(), "v": v.copy(), "w": w.copy()}]

    dot_int = max(1, n_steps // 20)
    if verbose:
        print("  Integrating: ", end="", flush=True)

    # ── Main time loop ────────────────────────────────────────────────────────
    for step in range(1, n_steps + 1):
        # Laplacians with Neumann BCs
        Lu = laplacian_neumann(u)
        Lv = laplacian_neumann(v)
        Lw = laplacian_neumann(w)

        # Clamped production terms
        F = F_prod(v, w, p)
        G = G_prod(u, w, p)
        H = H_prod(u, v, p)

        # Forward Euler update
        u += dt * (F - p["cu"]*u + p["Du"]*Lu)
        v += dt * (G - p["cv"]*v + p["Dv"]*Lv)
        w += dt * (H - p["cw"]*w + p["Dw"]*Lw)

        # Enforce physical non-negativity
        np.clip(u, 0.0, None, out=u)
        np.clip(v, 0.0, None, out=v)
        np.clip(w, 0.0, None, out=w)

        if step % save_every == 0:
            snapshots.append({"step": step, "time": step * dt,
                             "u": u.copy(), "v": v.copy(), "w": w.copy()})
        if verbose and step % dot_int == 0:
            print("▪", end="", flush=True)

    final = {"step": n_steps, "time": n_steps * dt,
             "u": u.copy(), "v": v.copy(), "w": w.copy()}

    if verbose:
        print(f"\n  Done.  "
              f"u∈[{u.min():.3f},{u.max():.3f}]  "
              f"v∈[{v.min():.3f},{v.max():.3f}]  "
              f"w∈[{w.min():.3f},{w.max():.3f}]")

    return {"params": p, "ss": ss, "snapshots": snapshots, "final": final}


# =============================================================================
# 8.  Visualisation helpers
# =============================================================================

def _norm01(x, lo_pct=2, hi_pct=98):
    """Percentile-clip and map to [0, 1]  (float32)."""
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    span = hi - lo
    if span < 1e-10:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32)


def fields_to_rgb(u, v):
    """
    Encode fields as an RGB image using the Kondo (2009) colour scheme:
      u  →  green channel  (melanophores)
      v  →  blue  channel  (xanthophores)

    Black  = both low   |  Green = high u, low v
    Blue   = low u, high v  |  Cyan = both high

    Returns (H, W, 3) float32 array with values in [0, 1].
    """
    rgb = np.zeros(u.shape + (3,), dtype=np.float32)
    rgb[..., 1] = _norm01(u)   # green ← melanophores
    rgb[..., 2] = _norm01(v)   # blue  ← xanthophores
    return rgb


def plot_fig4(sims, save_path=None, figsize=(15, 5.5)):
    """
    Reproduce Kondo (2009) Fig. 4.

    Parameters
    ----------
    sims      : list of 3 simulation-result dicts
                [sim_wildtype, sim_leopard, sim_jaguar]
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             gridspec_kw={"wspace": 0.05})
    for ax, sim in zip(axes, sims):
        ax.imshow(
            fields_to_rgb(sim["final"]["u"], sim["final"]["v"]),
            origin="lower", aspect="equal", interpolation="nearest",
        )
        ax.set_title(sim["params"]["name"],
                     fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")

    fig.legend(
        handles=[
            Patch(color=(0, 0.85, 0.0), label="high u — melanophores"),
            Patch(color=(0, 0.0,  0.9), label="high v — xanthophores"),
            Patch(color=(0, 0.55, 0.6), label="both high"),
            Patch(color="black",         label="both low"),
        ],
        loc="lower center", ncol=4, fontsize=10,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.suptitle(
        "Kondo (2009) Fig. 4 Reproduction\n"
        "128 × 128 grid  ·  Neumann BC  ·  explicit Euler",
        fontsize=13, y=1.03,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_time_evolution(sim, n_panels=6, save_path=None):
    """Show pattern emergence via n evenly-spaced snapshots."""
    snaps = sim["snapshots"]
    idxs  = np.linspace(0, len(snaps) - 1, n_panels, dtype=int)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(3.0 * n_panels, 3.3),
                             gridspec_kw={"wspace": 0.04})
    for ax, s in zip(axes, [snaps[i] for i in idxs]):
        ax.imshow(fields_to_rgb(s["u"], s["v"]),
                  origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(f"t = {s['time']:.0f}", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Time evolution — {sim['params']['name']}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_dispersion_relations(param_list, save_path=None):
    """
    Plot σ(k) for each parameter set, with the Turing-unstable band shaded.
    """
    n = len(param_list)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    colours = ["steelblue", "crimson", "seagreen"][:n]
    for ax, p, col in zip(axes, param_list, colours):
        res = stability_analysis(p, verbose=False)
        k, s = res["k"], res["sigma"]

        ax.plot(k, s, color=col, lw=2.2)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.fill_between(k, s, 0, where=s > 0,
                        color=col, alpha=0.5)