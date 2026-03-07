"""
HG_ATTN.py

Hypergraph-Guided Attention (HG-ATTN) for EEG leads/channels.

This module implements a practical, beginner-friendly version of the
"node -> hyperedge -> node" two-stage attention described in our discussion.

Key idea:
  - Nodes = EEG channels (leads). Optionally, you can add a virtual CLS node.
  - Hyperedges = prior channel groups (brain-region / hemisphere / symmetry / neighborhood).

We provide:
  1) A medically plausible prior hypergraph builder for BCICIV-2a (22 channels).
  2) A reusable HGAttnBlock (Transformer-like block) that can be plugged into models.

Author: generated with ChatGPT based on user's project context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Channel name preset: BCICIV-2a (22 EEG channels)
# (Order matches preprocess_reref.py / common BCIC-2a MAT format)
# =============================================================================
BCIC2A_CH_NAMES_22: List[str] = [
    "Fz",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2",
    "POz",
]


@dataclass
class HyperedgeSpec:
    """A named hyperedge with a semantic type and member node indices."""
    name: str
    edge_type: str
    members: List[int]


@dataclass
class HypergraphPriors:
    """
    Hypergraph priors to be used by HG-ATTN.

    membership_mask: (N_nodes, E) bool
      - True if node participates in the hyperedge for pooling (hyperedge token creation).
    incident_mask:   (N_nodes, E) bool
      - True if node is allowed to attend to that hyperedge (node -> hyperedge attention).
      - You may want incident_mask == membership_mask for normal nodes.
      - For a virtual CLS node, membership can be False but incident can be True (attend all edges).
    edge_names: list length E
    edge_types: list length E
    """
    membership_mask: torch.Tensor
    incident_mask: torch.Tensor
    edge_names: List[str]
    edge_types: List[str]


def _idx_map(ch_names: Sequence[str]) -> Dict[str, int]:
    return {str(name): i for i, name in enumerate(ch_names)}


def _safe_indices(ch_names: Sequence[str], names: Sequence[str]) -> List[int]:
    """Return indices for names that exist in ch_names."""
    m = _idx_map(ch_names)
    out: List[int] = []
    for n in names:
        if n in m:
            out.append(m[n])
    return out


def build_bcic2a_priors(
    ch_names: Optional[Sequence[str]] = None,
    include_region: bool = True,
    include_hemisphere: bool = True,
    include_midline: bool = True,
    include_symmetry: bool = True,
    include_neighborhood: bool = True,
    include_global: bool = True,
    add_virtual_cls: bool = True,
    virtual_cls_attend_all: bool = True,
) -> Tuple[HypergraphPriors, List[HyperedgeSpec]]:
    """
    Build a medically plausible prior hypergraph for BCICIV-2a.

    Notes for beginners:
      - These priors are NOT claiming true anatomical connectivity.
      - They encode "should be considered together" groups to help attention avoid collapsing to 1-2 channels.

    Parameters:
      ch_names:
        If None, defaults to BCIC2A_CH_NAMES_22.
        If you dropped a reference channel, pass the updated channel-name list so indices match your data.
      include_*:
        Choose which hyperedge families to include.
      add_virtual_cls:
        If True, we assume node 0 will be a virtual CLS node and channels start at index 1 in HG-ATTN input.
        We will build masks of shape (N_channels+1, E).
      virtual_cls_attend_all:
        If True, incident_mask[CLS, :] = True (CLS can attend all hyperedges).
        If False, CLS has no hyperedge attention (rarely useful).

    Returns:
      priors, edge_specs
    """
    if ch_names is None:
        ch_names = list(BCIC2A_CH_NAMES_22)
    else:
        ch_names = list(ch_names)

    Nch = len(ch_names)
    edges: List[HyperedgeSpec] = []

    # -------------------------
    # 1) Region / lobe-ish groups (coarse spatial priors)
    # -------------------------
    if include_region:
        frontal = _safe_indices(ch_names, ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4"])
        central = _safe_indices(ch_names, ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"])
        parietal = _safe_indices(ch_names, ["CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"])
        if len(frontal) >= 2:
            edges.append(HyperedgeSpec("Region_Frontal", "region", frontal))
        if len(central) >= 2:
            edges.append(HyperedgeSpec("Region_Central", "region", central))
        if len(parietal) >= 2:
            edges.append(HyperedgeSpec("Region_ParietalOcc", "region", parietal))

    # -------------------------
    # 2) Hemisphere groups
    # -------------------------
    if include_hemisphere:
        left = _safe_indices(ch_names, ["FC3", "FC1", "C5", "C3", "C1", "CP3", "CP1", "P1"])
        right = _safe_indices(ch_names, ["FC2", "FC4", "C2", "C4", "C6", "CP2", "CP4", "P2"])
        if len(left) >= 2:
            edges.append(HyperedgeSpec("Hemi_Left", "hemisphere", left))
        if len(right) >= 2:
            edges.append(HyperedgeSpec("Hemi_Right", "hemisphere", right))

    # -------------------------
    # 3) Midline group (Fz/FCz/Cz/CPz/Pz/POz)
    # -------------------------
    if include_midline:
        mid = _safe_indices(ch_names, ["Fz", "FCz", "Cz", "CPz", "Pz", "POz"])
        if len(mid) >= 2:
            edges.append(HyperedgeSpec("Midline", "midline", mid))

    # -------------------------
    # 4) Symmetry-inspired groups (left-right pairs + nearby anchors)
    # -------------------------
    if include_symmetry:
        # These are groups, not only pairs, to encourage "bilateral comparison" patterns.
        sym_fc = _safe_indices(ch_names, ["FC3", "FC4", "FC1", "FC2", "FCz", "Fz"])
        sym_c = _safe_indices(ch_names, ["C3", "C4", "C1", "C2", "Cz", "C5", "C6"])
        sym_cp = _safe_indices(ch_names, ["CP3", "CP4", "CP1", "CP2", "CPz"])
        sym_p = _safe_indices(ch_names, ["P1", "P2", "Pz", "POz", "CPz"])
        if len(sym_fc) >= 2:
            edges.append(HyperedgeSpec("Sym_FC", "symmetry", sym_fc))
        if len(sym_c) >= 2:
            edges.append(HyperedgeSpec("Sym_C", "symmetry", sym_c))
        if len(sym_cp) >= 2:
            edges.append(HyperedgeSpec("Sym_CP", "symmetry", sym_cp))
        if len(sym_p) >= 2:
            edges.append(HyperedgeSpec("Sym_P", "symmetry", sym_p))

    # -------------------------
    # 5) Neighborhood hyperedges (local spatial neighborhoods)
    # -------------------------
    if include_neighborhood:
        neigh_map = {
            "Fz":  ["FCz", "FC1", "FC2"],
            "FC3": ["FC1", "FCz", "C3", "C5", "Fz"],
            "FC1": ["FC3", "FCz", "C1", "Fz"],
            "FCz": ["Fz", "FC1", "FC2", "Cz", "C1", "C2"],
            "FC2": ["FCz", "FC4", "C2", "Fz"],
            "FC4": ["FC2", "FCz", "C4", "C6", "Fz"],
            "C5":  ["FC3", "C3", "CP3"],
            "C3":  ["C5", "FC3", "C1", "CP3"],
            "C1":  ["FC1", "C3", "Cz", "CP1"],
            "Cz":  ["FCz", "C1", "C2", "CPz"],
            "C2":  ["FC2", "Cz", "C4", "CP2"],
            "C4":  ["FC4", "C2", "C6", "CP4"],
            "C6":  ["FC4", "C4", "CP4"],
            "CP3": ["C3", "C5", "CP1", "P1"],
            "CP1": ["C1", "CP3", "CPz", "P1"],
            "CPz": ["Cz", "CP1", "CP2", "Pz", "POz"],
            "CP2": ["C2", "CPz", "CP4", "P2"],
            "CP4": ["C4", "C6", "CP2", "P2"],
            "P1":  ["CP1", "CP3", "Pz", "POz"],
            "Pz":  ["CPz", "P1", "P2", "POz"],
            "P2":  ["CP2", "CP4", "Pz", "POz"],
            "POz": ["CPz", "Pz", "P1", "P2"],
        }
        m = _idx_map(ch_names)
        for center_name, neigh_names in neigh_map.items():
            if center_name not in m:
                continue
            members = [m[center_name]] + [m[n] for n in neigh_names if n in m]
            # keep unique while preserving order
            seen = set()
            uniq = []
            for idx in members:
                if idx not in seen:
                    seen.add(idx)
                    uniq.append(idx)
            if len(uniq) >= 2:
                edges.append(HyperedgeSpec(f"Neigh_{center_name}", "neighborhood", uniq))

    # -------------------------
    # 6) Global hyperedge (all channels)
    # -------------------------
    if include_global and Nch >= 2:
        edges.append(HyperedgeSpec("Global_AllChannels", "global", list(range(Nch))))

    # Safety: ensure we have at least one edge
    if len(edges) == 0:
        edges.append(HyperedgeSpec("Global_AllChannels", "global", list(range(Nch))))

    # Build channel-level membership mask: (Nch, E)
    E = len(edges)
    mem_ch = torch.zeros((Nch, E), dtype=torch.bool)
    for e, spec in enumerate(edges):
        for i in spec.members:
            if 0 <= i < Nch:
                mem_ch[i, e] = True

    # Incident mask for channels: by default same as membership
    inc_ch = mem_ch.clone()

    # Extend masks with a virtual CLS node if requested
    if add_virtual_cls:
        N_nodes = Nch + 1
        mem = torch.zeros((N_nodes, E), dtype=torch.bool)
        inc = torch.zeros((N_nodes, E), dtype=torch.bool)
        # channel nodes are 1..Nch
        mem[1:, :] = mem_ch
        inc[1:, :] = inc_ch
        # CLS node is index 0
        if virtual_cls_attend_all:
            inc[0, :] = True
        else:
            inc[0, :] = False
        # CLS does not participate in pooling
        mem[0, :] = False
    else:
        mem = mem_ch
        inc = inc_ch

    priors = HypergraphPriors(
        membership_mask=mem,
        incident_mask=inc,
        edge_names=[e.name for e in edges],
        edge_types=[e.edge_type for e in edges],
    )
    return priors, edges


# =============================================================================
# HG-ATTN implementation (Transformer-like block)
# =============================================================================
class HyperedgePool(nn.Module):
    """
    Hyperedge pooling: create hyperedge tokens Z from node tokens X using a membership mask.

    Input:
      X: (B, N, D)
      membership_mask: (N, E) bool

    Output:
      Z: (B, E, D)
      P: (B, E, N) pooling weights (optional, mostly for interpretability)
    """

    def __init__(self, d_model: int, d_pool: int = 64, dropout: float = 0.0, pool: str = "attn"):
        super().__init__()
        self.d_model = int(d_model)
        self.d_pool = int(d_pool)
        self.pool = str(pool)
        self.dropout = nn.Dropout(float(dropout))

        if self.pool not in ("attn", "mean"):
            raise ValueError("pool must be 'attn' or 'mean'")

        # Attention pooling parameters
        if self.pool == "attn":
            self.Wp = nn.Linear(self.d_model, self.d_pool, bias=True)
            self.u = nn.Parameter(torch.zeros(self.d_pool))
            nn.init.normal_(self.u, std=0.02)

    def forward(
        self,
        X: torch.Tensor,
        membership_mask: torch.Tensor,
        return_p: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        X: (B,N,D)
        membership_mask: (N,E) bool
        """
        if X.dim() != 3:
            raise ValueError(f"X must be (B,N,D), got {tuple(X.shape)}")
        B, N, D = X.shape
        if D != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {D}")
        if membership_mask.dim() != 2:
            raise ValueError("membership_mask must be (N,E)")
        if membership_mask.shape[0] != N:
            raise ValueError(f"membership_mask N mismatch: {membership_mask.shape[0]} vs {N}")

        # If mean pooling: Z_e = average of member node features
        if self.pool == "mean":
            # membership_mask: (N,E) -> (E,N)
            m = membership_mask.transpose(0, 1).to(X.device)  # (E,N) bool
            m_f = m.float()  # (E,N)
            denom = m_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (E,1)
            # Z: (B,E,D) = (E,N) @ (B,N,D)
            Z = torch.einsum("en,bnd->bed", m_f / denom, X)
            return Z, None if not return_p else (m_f / denom).unsqueeze(0).expand(B, -1, -1)

        # Attention pooling: edge-wise softmax over member nodes
        # score per node: s_j = u^T tanh(Wp X_j)
        # scores_node: (B,N)
        scores_node = torch.einsum("bnd,d->bn", torch.tanh(self.Wp(X)), self.u)

        # Expand to (B,E,N) and mask non-members
        m = membership_mask.transpose(0, 1).to(X.device)  # (E,N) bool
        scores_en = scores_node.unsqueeze(1).expand(B, m.shape[0], N)  # (B,E,N)
        scores_en = scores_en.masked_fill(~m.unsqueeze(0), float("-inf"))

        # Softmax over N (within each hyperedge)
        P = torch.softmax(scores_en, dim=-1)  # (B,E,N)
        P = self.dropout(P)

        Z = torch.einsum("ben,bnd->bed", P, X)  # (B,E,D)

        if return_p:
            return Z, P
        return Z, None


class HGAttnBlock(nn.Module):
    """
    Hypergraph-Guided Attention block (node -> hyperedge -> node), Transformer-style.

    Input:
      X: (B, N, D)  node tokens (channels [+ optional CLS])
      membership_mask: (N, E) bool (for hyperedge pooling)
      incident_mask:   (N, E) bool (for node->edge attention)

    Output:
      X_out: (B, N, D)
      attn:  (B, H, N, E) if return_attn else None
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        pool: str = "attn",
        d_pool: int = 64,
        gate: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout_p = float(dropout)
        self.gate = bool(gate)

        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.dk = self.d_model // self.num_heads
        self.dv = self.dk

        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        self.pool = HyperedgePool(d_model=self.d_model, d_pool=d_pool, dropout=dropout, pool=pool)

        self.Wq = nn.Linear(self.d_model, self.d_model, bias=True)
        self.Wk = nn.Linear(self.d_model, self.d_model, bias=True)
        self.Wv = nn.Linear(self.d_model, self.d_model, bias=True)
        self.Wo = nn.Linear(self.d_model, self.d_model, bias=True)

        self.drop_attn = nn.Dropout(self.dropout_p)

        if self.gate:
            self.Wg = nn.Linear(2 * self.d_model, self.d_model, bias=True)

        self.ffn1 = nn.Linear(self.d_model, dim_feedforward)
        self.ffn2 = nn.Linear(dim_feedforward, self.d_model)
        self.drop_ffn = nn.Dropout(self.dropout_p)

        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        else:
            raise ValueError("activation must be gelu or relu")

    def forward(
        self,
        X: torch.Tensor,
        membership_mask: torch.Tensor,
        incident_mask: torch.Tensor,
        return_attn: bool = False,
        return_pool: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          X_out
          attn (optional): (B,H,N,E)
          pool_weights (optional): (B,E,N)
        """
        if X.dim() != 3:
            raise ValueError(f"X must be (B,N,D), got {tuple(X.shape)}")
        B, N, D = X.shape
        if D != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {D}")
        if membership_mask.shape[0] != N or incident_mask.shape[0] != N:
            raise ValueError("membership_mask/incident_mask first dim must match N")
        if membership_mask.shape[1] != incident_mask.shape[1]:
            raise ValueError("membership_mask/incident_mask must have same E")
        E = membership_mask.shape[1]

        # Pre-LN
        Xn = self.ln1(X)

        # Hyperedge tokens
        Z, P = self.pool(Xn, membership_mask=membership_mask, return_p=return_pool)  # Z: (B,E,D)

        # Project
        Q = self.Wq(Xn).view(B, N, self.num_heads, self.dk).transpose(1, 2)  # (B,H,N,dk)
        K = self.Wk(Z).view(B, E, self.num_heads, self.dk).transpose(1, 2)   # (B,H,E,dk)
        V = self.Wv(Z).view(B, E, self.num_heads, self.dv).transpose(1, 2)   # (B,H,E,dv)

        # Attention scores: (B,H,N,E)
        scores = torch.einsum("bhnd,bhed->bhne", Q, K) / (self.dk ** 0.5)

        # Apply incident mask
        inc = incident_mask.to(X.device).unsqueeze(0).unsqueeze(0)  # (1,1,N,E)
        scores = scores.masked_fill(~inc, float("-inf"))

        A = torch.softmax(scores, dim=-1)  # (B,H,N,E)
        A = self.drop_attn(A)

        # Message: (B,H,N,dv)
        M = torch.einsum("bhne,bhed->bhnd", A, V)
        M = M.transpose(1, 2).contiguous().view(B, N, self.d_model)  # merge heads

        M = self.Wo(M)
        M = self.drop_ffn(M)

        # Residual + (optional) gate
        if self.gate:
            G = torch.sigmoid(self.Wg(torch.cat([Xn, M], dim=-1)))
            Y = X + G * M
        else:
            Y = X + M

        # FFN
        Yn = self.ln2(Y)
        FF = self.ffn2(self.drop_ffn(self.act(self.ffn1(Yn))))
        X_out = Y + self.drop_ffn(FF)

        if return_attn and return_pool:
            return X_out, A, P
        if return_attn:
            return X_out, A, None
        if return_pool:
            return X_out, None, P
        return X_out, None, None
