import torch
from test.evaluation.metrics import identity


def decoder_function(model, kind):
    if kind == "gpt":
        return lambda tokens, memory: model.hopping_decoder(
            tokens, None, memory.clone()
        )[0]
    if kind != "crossattn":
        raise ValueError(kind)

    def forward(tokens, memory):
        mask = torch.ones(
            tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool
        ).triu(1)
        return model.smiles_decoder(tokens, memory, mask, tokens.eq(0))

    return forward


@torch.no_grad()
def sample_sequence(
    logits_fn,
    memory,
    vocab,
    max_new_tokens=79,
    temperature=1.0,
    top_p=0.9,
    generator=None,
):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling parameters")
    tokens = torch.tensor([[vocab["BOS"]]], device=memory.device)
    log_probability = 0.0
    terminated = False
    for step in range(max_new_tokens):
        logits = logits_fn(tokens, memory.clone())[:, -1].clone() / temperature
        logits[:, [vocab["PAD"], vocab["BOS"]]] = -torch.inf
        ordered, indices = torch.sort(logits, descending=True, dim=-1)
        remove = ordered.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        ordered[remove] = -torch.inf
        probabilities = ordered.softmax(-1)
        choice = torch.multinomial(probabilities, 1, generator=generator)
        token = indices.gather(-1, choice)
        log_probability += float(probabilities.gather(-1, choice).log())
        tokens = torch.cat([tokens, token], dim=1)
        if token.item() == vocab["EOS"]:
            terminated = True
            break
    inv = {v: k for k, v in vocab.items()}
    ids = tokens[0].tolist()
    text = "".join(inv[t] for t in ids[1:] if t not in (vocab["EOS"], vocab["PAD"]))
    return dict(
        generated_smiles=text,
        token_ids=ids,
        log_probability=log_probability,
        terminated_eos=terminated,
        **identity(text),
    )


@torch.inference_mode()
def completed_beams(
    logits_fn, memory, width=20, limit=20, max_new_tokens=79, penalty=1.0, vocab=None
):
    """Return length-normalized finished beams, preserving log-space scores."""
    if min(width, limit, max_new_tokens) < 1 or penalty < 0:
        raise ValueError("Invalid search settings")
    if vocab is None:
        raise ValueError("Supply the exact checkpoint vocabulary")
    VOCAB = vocab
    bos, eos, pad = (VOCAB["BOS"], VOCAB["EOS"], VOCAB["PAD"])
    tokens = torch.tensor([[bos]], device=memory.device)
    scores = torch.zeros(1, device=memory.device)
    finished = []
    original = memory.clone()
    for step in range(max_new_tokens):
        logits = logits_fn(tokens, memory.expand(len(tokens), -1, -1).clone())[
            :, -1
        ].clone()
        logits[:, [pad, bos]] = -torch.inf
        if step == 0:
            logits[:, eos] = -torch.inf
        logp = torch.log_softmax(logits, dim=-1)
        per_parent = min(width + 1, logp.shape[-1])
        vals, ids = torch.topk(logp, per_parent, dim=-1)
        cumulative = vals + scores[:, None]
        ordering = torch.argsort(cumulative.flatten(), descending=True).tolist()
        values, next_ids = (cumulative.cpu().tolist(), ids.cpu().tolist())
        old_tokens = tokens.tolist()
        live_tokens, live_scores = ([], [])
        for flat_index in ordering:
            parent, child = divmod(flat_index, per_parent)
            score, token = (values[parent][child], next_ids[parent][child])
            if score == float("-inf"):
                continue
            sequence = old_tokens[parent] + [token]
            if token == eos:
                finished.append(
                    dict(
                        token_ids=sequence,
                        log_probability=score,
                        generated_length=len(sequence) - 1,
                        beam_score=score / (len(sequence) - 1) ** penalty,
                    )
                )
            else:
                live_tokens.append(sequence)
                live_scores.append(score)
            if len(live_tokens) == width:
                break
        finished.sort(key=lambda x: (-x["beam_score"], x["token_ids"]))
        finished = finished[:limit]
        if not live_tokens:
            break
        tokens = torch.tensor(live_tokens, device=memory.device)
        scores = torch.tensor(live_scores, device=memory.device)
        optimistic = max(live_scores) / max_new_tokens**penalty
        if len(finished) >= limit and optimistic <= finished[-1]["beam_score"]:
            break
    if not torch.equal(memory, original):
        raise RuntimeError("Beam decoding changed the spectrum condition")
    inv = {v: k for k, v in VOCAB.items()}
    for rank, record in enumerate(finished, 1):
        record["beam_rank"] = rank
        record["generated_smiles"] = "".join(
            (inv[t] for t in record["token_ids"][1:-1])
        )
        record.update(identity(record["generated_smiles"]))
        record["terminated_eos"] = True
    return finished
