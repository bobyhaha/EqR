from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math

import torch
from pydantic import BaseModel
from torch import nn
import torch.nn.functional as F

from models.common import trunc_normal_init_
from models.layers import Attention, CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding, RotaryEmbedding2D, SwiGLU, rms_norm


@dataclass
class LatentCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class ModelCarry:
    inner_carry: LatentCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    global_step: Optional[int] = None
    halt_counter: Optional[torch.Tensor] = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class LGPRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    hidden_size: int
    num_heads: int = 1
    expansion: float = 4.0
    forward_dtype: str = "bfloat16"
    halt_max_steps: int
    halt_exploration_prob: float
    halt_threshold: float = 0.0

    n_explorers: int = 8
    explorer_mult: float = 0.5
    pi_layers: int = 2
    lg_steps: int = 4
    library_size: int = 256
    rag_library_size: Optional[int] = None
    mlp_library_mult: float = 4.0
    gate_mode: str = "hard"
    gate_threshold: float = 0.5
    straight_through_gate: bool = True
    forced_library: bool = False
    use_library: bool = True
    proposal_pool: str = "concat"
    recurrent_update: str = "residual"
    workspace: str = "token"
    pos_encodings: Optional[str] = "rope2d"
    rope_theta: float = 10000.0
    board_height: Optional[int] = None
    board_width: Optional[int] = None

    phd_lambda: float = 0.95
    phd_noise_scale: float = 0.0
    H_init_std: float = 1.0
    L_init_std: float = 1.0
    rms_norm_eps: float = 1e-5


class Explorer(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        h = max(8, int(d * config.explorer_mult))
        self.up = CastedLinear(2 * d, h, bias=True)
        self.down = CastedLinear(h, d, bias=True)

    def forward(self, state: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, xs], dim=-1)
        return self.down(F.gelu(self.up(rms_norm(x, self.config.rms_norm_eps))))


class ExplorerBank(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Explorer(config) for _ in range(config.n_explorers)])
        self.lambda_ = float(config.phd_lambda)
        self.noise_scale = float(config.phd_noise_scale)

    def forward(self, state: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        proposals = torch.stack([layer(state, xs) for layer in self.layers], dim=1)
        if self.noise_scale <= 0:
            return proposals
        updated = proposals
        noise = torch.randn_like(updated) * self.noise_scale
        return (1.0 - self.lambda_) * proposals.detach() + self.lambda_ * updated + noise


class RAGLibrary(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        size = config.rag_library_size or config.library_size
        d = config.hidden_size
        self.keys = nn.Parameter(torch.randn(size, d) / math.sqrt(d))
        self.values = nn.Parameter(torch.randn(size, d) / math.sqrt(d))

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.keys.to(q.dtype)
        values = self.values.to(q.dtype)
        attn = F.softmax(q @ keys.T / math.sqrt(q.shape[-1]), dim=-1)
        return attn @ values, attn


class MLPLibrary(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        h = max(8, int(d * config.mlp_library_mult))
        self.up = CastedLinear(d, h, bias=True)
        self.down = CastedLinear(h, d, bias=True)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        q = rms_norm(q, self.config.rms_norm_eps)
        return self.down(F.gelu(self.up(q)))


class Gate(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.up = CastedLinear(2 * d, d, bias=True)
        self.down = CastedLinear(d, 1, bias=True)

    def forward(self, state: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, xs], dim=-1)
        return torch.sigmoid(self.down(F.gelu(self.up(rms_norm(x, self.config.rms_norm_eps)))))


class PIBlock(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.config = config
        self.self_attn = Attention(
            hidden_size=d,
            head_dim=d // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(d, config.expansion)

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), self.config.rms_norm_eps)
        return rms_norm(hidden_states + self.mlp(hidden_states), self.config.rms_norm_eps)


class PIModule(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        proposal_width = config.n_explorers * d if config.proposal_pool.lower() == "concat" else d
        self.in_proj = CastedLinear(3 * d + proposal_width, d, bias=True)
        self.layers = nn.ModuleList([PIBlock(config) for _ in range(config.pi_layers)])
        self.out = CastedLinear(d, d, bias=True)

    def forward(self, state: torch.Tensor, xs: torch.Tensor, proposal: torch.Tensor, libvec: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        z = self.in_proj(torch.cat([state, xs, proposal, libvec], dim=-1))
        for layer in self.layers:
            z = layer(z, cos_sin)
        return state + self.out(rms_norm(z, self.config.rms_norm_eps))


class InnerNetwork(nn.Module):
    def __init__(self, config: LGPRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        d = config.hidden_size
        self.embed_scale = math.sqrt(d)
        self.embed_tokens = CastedEmbedding(config.vocab_size, d, 1.0 / self.embed_scale, self.forward_dtype)
        self.state_init = CastedLinear(d, d, bias=True)
        self.explorers = ExplorerBank(config)
        self.rag_library = RAGLibrary(config) if config.use_library else None
        self.mlp_library = MLPLibrary(config) if config.use_library else None
        self.gate = Gate(config) if config.use_library else None
        self.pi = PIModule(config)
        self._init_pos()
        self.lm_head = CastedLinear(d, config.vocab_size, bias=False)
        self.q_head = CastedLinear(d, 2, bias=True)
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(d, dtype=self.forward_dtype), std=config.H_init_std), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(d, dtype=self.forward_dtype), std=config.L_init_std), persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _board_dims(self) -> Tuple[int, int]:
        if self.config.board_height is not None and self.config.board_width is not None:
            return self.config.board_height, self.config.board_width
        board_size = int(self.config.seq_len**0.5)
        if board_size * board_size != self.config.seq_len:
            raise ValueError(f"seq_len {self.config.seq_len} is not a perfect square. Specify board_height and board_width explicitly.")
        return board_size, board_size

    def _init_pos(self) -> None:
        pos = self.config.pos_encodings
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.config.hidden_size} must be divisible by num_heads={self.config.num_heads}"
            )
        head_dim = self.config.hidden_size // self.config.num_heads
        if pos == "rope" and head_dim % 2 != 0:
            raise ValueError(
                f"rope requires an even attention head_dim; got hidden_size={self.config.hidden_size}, "
                f"num_heads={self.config.num_heads}, head_dim={head_dim}"
            )
        if pos == "rope2d" and head_dim % 4 != 0:
            raise ValueError(
                f"rope2d requires attention head_dim divisible by 4; got hidden_size={self.config.hidden_size}, "
                f"num_heads={self.config.num_heads}, head_dim={head_dim}"
            )
        if pos == "rope":
            self.rotary_emb = RotaryEmbedding(head_dim, self.config.seq_len, self.config.rope_theta)
        elif pos == "rope2d":
            h, w = self._board_dims()
            self.rotary_emb = RotaryEmbedding2D(head_dim, h, w, self.config.rope_theta)
        elif pos not in {"none", None}:
            raise ValueError(f"Unknown pos_encodings '{pos}'")

    def _cos_sin(self) -> CosSin:
        return self.rotary_emb() if hasattr(self, "rotary_emb") else None

    def _pool_proposals(self, state: torch.Tensor, proposals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = self.config.proposal_pool.lower()
        if proposals.ndim != 4:
            raise ValueError("LG-PRM requires token workspace proposals shaped [B, n_explorers, L, d]")
        if mode == "concat":
            b, n, l, d = proposals.shape
            attn = proposals.new_full(proposals.shape[:-1], 1.0 / n)
            return proposals.permute(0, 2, 1, 3).reshape(b, l, n * d), attn
        if mode == "attention":
            query = rms_norm(state, self.config.rms_norm_eps).unsqueeze(1)
            keys = rms_norm(proposals, self.config.rms_norm_eps)
            scores = (query * keys).sum(dim=-1) / math.sqrt(proposals.shape[-1])
            attn = F.softmax(scores, dim=1)
            return (attn.unsqueeze(-1) * proposals).sum(dim=1), attn
        raise ValueError(f"Unknown proposal_pool: {self.config.proposal_pool}")

    def _proposal_diversity(self, proposals: torch.Tensor) -> torch.Tensor:
        if proposals.shape[1] <= 1:
            return proposals.new_tensor(0.0, dtype=torch.float32)
        p = F.normalize(proposals.to(torch.float32), dim=-1)
        if proposals.ndim == 4:
            p = p.permute(0, 2, 1, 3)
            sim = p @ p.transpose(-1, -2)
        else:
            sim = p @ p.transpose(1, 2)
        eye = torch.eye(sim.shape[-1], dtype=torch.bool, device=sim.device)
        eye = eye.view(1, 1, sim.shape[-1], sim.shape[-1]) if sim.ndim == 4 else eye.unsqueeze(0)
        off_diag = sim.masked_fill(eye, 0.0)
        denom = proposals.shape[1] * (proposals.shape[1] - 1)
        return (off_diag.square().sum(dim=(-1, -2)) / denom).mean()

    def _proposal_usage_stats(self, proposal_attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reduce_dims = (0, 2) if proposal_attn.ndim == 3 else 0
        usage = proposal_attn.to(torch.float32).mean(dim=reduce_dims)
        usage_entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
        n_explorers = proposal_attn.shape[1]
        effective_used = usage_entropy.exp().clamp(max=float(n_explorers))
        load_balance = n_explorers * usage.square().sum() - 1.0
        return effective_used, load_balance

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_scale * self.embed_tokens(input_ids.to(torch.int32))

    def empty_carry(self, batch_size: int, device: Optional[torch.device] = None) -> LatentCarry:
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        return LatentCarry(
            z_H=torch.empty(shape, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(shape, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: LatentCarry) -> LatentCarry:
        h = self.H_init.to(device=carry.z_H.device, dtype=carry.z_H.dtype).view(1, 1, -1)
        l = self.L_init.to(device=carry.z_L.device, dtype=carry.z_L.dtype).view(1, 1, -1)
        mask = reset_flag.to(device=carry.z_H.device).view(-1, 1, 1)
        return LatentCarry(torch.where(mask, h, carry.z_H), torch.where(mask, l, carry.z_L))

    def _gate_value(self, gate_prob: torch.Tensor) -> torch.Tensor:
        if self.config.forced_library:
            return torch.ones_like(gate_prob)
        if self.config.gate_mode == "soft":
            return gate_prob
        hard = (gate_prob >= self.config.gate_threshold).to(gate_prob.dtype)
        if self.training and self.config.straight_through_gate:
            return hard + gate_prob - gate_prob.detach()
        return hard

    def forward_no_carry(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        token_features = self._input_embeddings(batch["inputs"])
        xs = token_features
        carried_tokens = rms_norm(z_H + z_L + token_features, self.config.rms_norm_eps)
        state = self.state_init(carried_tokens)
        gate_values = []
        hard_values = []
        entropies = []
        proposal_entropies = []
        proposal_diversities = []
        effective_phds = []
        load_balance_losses = []
        cos_sin = self._cos_sin()
        for _ in range(self.config.lg_steps):
            proposals = self.explorers(state, xs)
            proposal, proposal_attn = self._pool_proposals(state, proposals)
            proposal_entropies.append((-(proposal_attn * proposal_attn.clamp_min(1e-8).log()).sum(dim=1)).to(torch.float32).mean())
            proposal_diversities.append(self._proposal_diversity(proposals))
            effective_used, load_balance = self._proposal_usage_stats(proposal_attn)
            effective_phds.append(effective_used)
            load_balance_losses.append(load_balance)
            if self.config.use_library:
                gate_prob = self.gate(state, xs)
                gate = self._gate_value(gate_prob)
                rag, attn = self.rag_library(state)
                mlp = self.mlp_library(state)
                libvec = gate * 0.5 * (rag + mlp)
                gate_values.append(gate_prob.to(torch.float32).mean())
                hard_values.append((gate >= self.config.gate_threshold).to(torch.float32).mean())
                entropies.append((-(attn * attn.clamp_min(1e-8).log()).sum(dim=-1)).to(torch.float32).mean())
            else:
                libvec = torch.zeros_like(state)
                gate_values.append(state.new_tensor(0.0, dtype=torch.float32))
                hard_values.append(state.new_tensor(0.0, dtype=torch.float32))
                entropies.append(state.new_tensor(0.0, dtype=torch.float32))
            state = self.pi(state, xs, proposal, libvec, cos_sin)
        diagnostics = {
            "gate_mean": torch.stack(gate_values).mean() if gate_values else state.new_tensor(0.0, dtype=torch.float32),
            "hard_gate_mean": torch.stack(hard_values).mean() if hard_values else state.new_tensor(0.0, dtype=torch.float32),
            "library_entropy": torch.stack(entropies).mean() if entropies else state.new_tensor(0.0, dtype=torch.float32),
            "proposal_entropy": torch.stack(proposal_entropies).mean() if proposal_entropies else state.new_tensor(0.0, dtype=torch.float32),
            "proposal_diversity_loss": torch.stack(proposal_diversities).mean() if proposal_diversities else state.new_tensor(0.0, dtype=torch.float32),
            "effective_phds_used": torch.stack(effective_phds).mean() if effective_phds else state.new_tensor(0.0, dtype=torch.float32),
            "proposal_load_balance_loss": torch.stack(load_balance_losses).mean() if load_balance_losses else state.new_tensor(0.0, dtype=torch.float32),
        }
        if self.config.recurrent_update.lower() == "legacy":
            z_H = token_features + state
        elif self.config.recurrent_update.lower() == "residual":
            z_H = rms_norm(carried_tokens + state, self.config.rms_norm_eps)
        else:
            raise ValueError(f"Unknown recurrent_update: {self.config.recurrent_update}")
        z_L = rms_norm(z_L + state + token_features, self.config.rms_norm_eps)
        logits = self.lm_head(rms_norm(z_H, self.config.rms_norm_eps))
        q_state = state[:, 0] if state.ndim == 3 else state
        q = self.q_head(rms_norm(q_state, self.config.rms_norm_eps)).to(torch.float32)
        return (z_H, z_L), logits, (q[..., 0], q[..., 1]), diagnostics

    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        (z_H, z_L), logits, q, diagnostics = self.forward_no_carry(carry.z_H, carry.z_L, batch)
        return LatentCarry(z_H.detach(), z_L.detach()), logits, q, diagnostics

    def concat_states(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        return torch.cat((z_H, z_L), dim=1)

    def split_states(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.config.seq_len
        return state[:, :n, :], state[:, n:, :]


class LGPRMModel(nn.Module):
    gate_mode: str = "hard"
    phd_noise_scale: float = 0.0

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config_dict = dict(config_dict)
        config_dict.setdefault("gate_mode", self.gate_mode)
        config_dict.setdefault("phd_noise_scale", self.phd_noise_scale)
        self.config = LGPRMConfig(**config_dict)
        self.inner = InnerNetwork(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ModelCarry:
        b, device = batch["inputs"].shape[0], batch["inputs"].device
        c = self.inner.empty_carry(b)
        return ModelCarry(
            inner_carry=LatentCarry(c.z_H.to(device), c.z_L.to(device)),
            steps=torch.zeros((b,), dtype=torch.int32, device=device),
            halted=torch.ones((b,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
            halt_counter=torch.zeros((b,), dtype=torch.int32, device=device),
        )

    def forward(self, carry: ModelCarry, batch: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        del kwargs
        device = batch["inputs"].device
        halted_prev = carry.halted.to(device)
        inner = self.inner.reset_carry(halted_prev, LatentCarry(carry.inner_carry.z_H.to(device), carry.inner_carry.z_L.to(device)))
        steps = torch.where(halted_prev, torch.zeros_like(carry.steps.to(device)), carry.steps.to(device))
        data = {
            k: torch.where(halted_prev.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v.to(device))
            for k, v in carry.current_data.items()
        }
        inner, logits, (q_halt, q_continue), diagnostics = self.inner(inner, data)
        out = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_continue,
            **diagnostics,
        }
        with torch.no_grad():
            steps = steps + 1
            is_last = steps >= self.config.halt_max_steps
            halted = is_last
            if self.training and self.config.halt_max_steps > 1:
                halted = halted | (q_halt > float(self.config.halt_threshold))
                min_steps = (torch.rand_like(q_halt) < self.config.halt_exploration_prob) * torch.randint_like(steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (steps >= min_steps)
        return ModelCarry(inner, steps, halted, data, halt_counter=torch.zeros_like(steps)), out

    def pack_solver_state(self, z_H: torch.Tensor, z_L: torch.Tensor) -> torch.Tensor:
        return self.inner.concat_states(z_H, z_L)

    def unpack_solver_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inner.split_states(state)

    def initial_solver_state(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        b, device = batch["inputs"].shape[0], batch["inputs"].device
        shape = (b, self.config.seq_len, self.config.hidden_size)
        z_H = self.inner.H_init.to(device=device).view(1, 1, -1).expand(shape).clone()
        z_L = self.inner.L_init.to(device=device).view(1, 1, -1).expand(shape).clone()
        return self.pack_solver_state(z_H, z_L)

    def solver_step(self, state: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z_H, z_L = self.unpack_solver_state(state)
        (z_H, z_L), logits, q, _ = self.inner.forward_no_carry(z_H, z_L, batch)
        return self.pack_solver_state(z_H.detach(), z_L.detach()), logits, q


class LGPRMHardModel(LGPRMModel):
    gate_mode = "hard"
    phd_noise_scale = 0.0


class LGPRMSoftModel(LGPRMModel):
    gate_mode = "soft"
    phd_noise_scale = 0.0


class LGPRMNoisyHardModel(LGPRMModel):
    gate_mode = "hard"
    phd_noise_scale = 0.01


class LGPRMNoisySoftModel(LGPRMModel):
    gate_mode = "soft"
    phd_noise_scale = 0.01


class LGPRMNoLibraryModel(LGPRMModel):
    gate_mode = "hard"
    phd_noise_scale = 0.0

    def __init__(self, config_dict: dict) -> None:
        config_dict = dict(config_dict)
        config_dict["use_library"] = False
        super().__init__(config_dict)


class LGPRMConcatModel(LGPRMModel):
    gate_mode = "hard"
    phd_noise_scale = 0.0

    def __init__(self, config_dict: dict) -> None:
        config_dict = dict(config_dict)
        config_dict["proposal_pool"] = "concat"
        super().__init__(config_dict)


class LGPRMAttentionModel(LGPRMModel):
    gate_mode = "hard"
    phd_noise_scale = 0.0

    def __init__(self, config_dict: dict) -> None:
        config_dict = dict(config_dict)
        config_dict["proposal_pool"] = "attention"
        super().__init__(config_dict)
