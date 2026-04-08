from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from poetry_lm.tokenizer import STRUCTURED_MODE, Tokenizer, control_tokens, structured_window_to_training_text


@dataclass
class RefinerConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    bias: bool = True


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config: RefinerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class RefinerMLP(nn.Module):
    def __init__(self, config: RefinerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class RefinerBlock(nn.Module):
    def __init__(self, config: RefinerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = RefinerMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MaskedRefiner(nn.Module):
    def __init__(self, config: RefinerConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([RefinerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} > block size {self.config.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            flat_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            )
            if loss_mask is not None:
                weights = loss_mask.view(-1).float()
                denom = torch.clamp(weights.sum(), min=1.0)
                loss = (flat_loss * weights).sum() / denom
            else:
                loss = flat_loss.mean()
        return logits, loss


def _sample_top_k(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    forbidden_ids: set[int] | None = None,
) -> int:
    scaled = logits / max(temperature, 1e-5)
    if forbidden_ids:
        scaled = scaled.clone()
        for token_id in forbidden_ids:
            if 0 <= token_id < scaled.size(-1):
                scaled[token_id] = -float("inf")
    if top_k is not None:
        values, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        scaled = scaled.clone()
        scaled[scaled < values[-1]] = -float("inf")
    probs = F.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def line_content_positions(ids: list[int], tokenizer: Tokenizer) -> dict[int, list[int]]:
    control = set(control_tokens())
    control.add("<MASK>")
    lines = {idx: [] for idx in range(1, 9)}
    current_line: int | None = None
    for pos, token_id in enumerate(ids):
        if token_id in {tokenizer.bos_id, tokenizer.eos_id}:
            continue
        piece = tokenizer.piece(int(token_id))
        if piece.startswith("<L") and piece.endswith(">"):
            try:
                current_line = int(piece[2:-1])
            except ValueError:
                current_line = None
            continue
        if piece in control:
            if piece == "<NL>":
                current_line = None
            continue
        if current_line is not None and current_line in lines:
            lines[current_line].append(pos)
    return lines


def candidate_refine_positions(
    ids: list[int],
    tokenizer: Tokenizer,
    tail_span: int = 3,
    tail_boost: float = 3.0,
) -> tuple[list[int], torch.Tensor]:
    by_line = line_content_positions(ids, tokenizer)
    candidates: list[int] = []
    weights: list[float] = []
    for line_no in range(2, 9):
        positions = by_line.get(line_no, [])
        if not positions:
            continue
        tail = set(positions[-tail_span:]) if line_no in {2, 4, 6, 8} else set()
        for pos in positions:
            candidates.append(pos)
            weights.append(tail_boost if pos in tail else 1.0)
    return candidates, torch.tensor(weights, dtype=torch.float32)


def corrupt_ids(
    ids: list[int],
    tokenizer: Tokenizer,
    mask_prob: float = 0.18,
    tail_span: int = 3,
    tail_boost: float = 3.0,
) -> tuple[list[int], list[int], list[int]]:
    mask_id = tokenizer.mask_id()
    if mask_id < 0 or mask_id == tokenizer.sp.unk_id():
        raise ValueError("tokenizer does not define a dedicated <MASK> token")
    candidates, weights = candidate_refine_positions(ids, tokenizer, tail_span=tail_span, tail_boost=tail_boost)
    if not candidates:
        return ids[:], ids[:], [0] * len(ids)
    n_mask = min(max(1, round(len(candidates) * mask_prob)), len(candidates))
    picked = torch.multinomial(weights, n_mask, replacement=False).tolist()
    masked_positions = {candidates[idx] for idx in picked}
    input_ids = ids[:]
    targets = ids[:]
    loss_mask = [0] * len(ids)
    for pos in masked_positions:
        input_ids[pos] = mask_id
        loss_mask[pos] = 1
    return input_ids, targets, loss_mask


def load_refiner(checkpoint_path: str | Path, tokenizer_model: str | Path, device: str = "cpu") -> tuple[MaskedRefiner, Tokenizer, dict]:
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    tokenizer = Tokenizer(tokenizer_model)
    model = MaskedRefiner(RefinerConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, ckpt


@torch.no_grad()
def refine_ids(
    model: MaskedRefiner,
    tokenizer: Tokenizer,
    ids: list[int],
    device: str,
    steps: int = 8,
    temperature: float = 0.8,
    top_k: int = 32,
    remask_ratio_start: float = 0.35,
    remask_ratio_end: float = 0.08,
    tail_span: int = 3,
    tail_boost: float = 3.0,
) -> list[int]:
    current = ids[:]
    candidates, _ = candidate_refine_positions(current, tokenizer, tail_span=tail_span, tail_boost=tail_boost)
    if not candidates:
        return current
    forbidden_ids = {
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.sp.pad_id(),
        tokenizer.mask_id(),
    }
    for piece in control_tokens():
        token_id = tokenizer.piece_id(piece)
        if token_id >= 0:
            forbidden_ids.add(token_id)
    for step in range(steps):
        x = torch.tensor([current], dtype=torch.long, device=device)
        logits, _ = model(x)
        token_logits = logits[0]
        conf = F.softmax(token_logits[candidates], dim=-1).gather(
            1,
            torch.tensor([current[pos] for pos in candidates], dtype=torch.long, device=device).unsqueeze(1),
        ).squeeze(1)
        progress = step / max(steps - 1, 1)
        remask_ratio = remask_ratio_start + (remask_ratio_end - remask_ratio_start) * progress
        n_remask = min(max(1, round(len(candidates) * remask_ratio)), len(candidates))
        low_conf_idx = torch.topk(-conf, k=n_remask).indices.tolist()
        remask_positions = [candidates[idx] for idx in low_conf_idx]

        masked = current[:]
        for pos in remask_positions:
            masked[pos] = tokenizer.mask_id()

        masked_x = torch.tensor([masked], dtype=torch.long, device=device)
        masked_logits, _ = model(masked_x)
        masked_token_logits = masked_logits[0]
        for pos in remask_positions:
            current[pos] = _sample_top_k(
                masked_token_logits[pos],
                temperature=temperature,
                top_k=top_k,
                forbidden_ids=forbidden_ids,
            )
    return current


@torch.no_grad()
def refine_draft_text(
    model: MaskedRefiner,
    tokenizer: Tokenizer,
    draft_text: str,
    device: str,
    steps: int = 8,
    temperature: float = 0.8,
    top_k: int = 32,
) -> str:
    lines = [line.strip() for line in draft_text.splitlines() if line.strip()]
    if len(lines) < 8:
        return "\n".join(lines)
    formatted = structured_window_to_training_text(lines[:8])
    ids = tokenizer.encode_formatted(formatted)
    refined_ids = refine_ids(
        model=model,
        tokenizer=tokenizer,
        ids=ids,
        device=device,
        steps=steps,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(refined_ids, mode=STRUCTURED_MODE)
