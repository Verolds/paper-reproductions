"""
Los tres experimentos que convierten la implementación en una reproducción.

  1. CONTEXTO  — crece n. El n-grama colapsa por dispersión, el MLP no.
                 Ésta es la tesis del paper (la "maldición de la dimensionalidad").
  2. CAPACIDAD — crece el modelo. Muestra que el modelo base está underfitting,
                 no "sin overfitting".
  3. EMBEDDINGS— entrena con m=2 y dibuja los 27 caracteres. El mecanismo del
                 paper hecho imagen: caracteres que se comportan parecido quedan cerca.

    python analysis.py           # los tres (~10 min: entrena 13 modelos)
    python analysis.py contexto  # solo uno
    python analysis.py replot    # re-dibuja desde analysis.json, sin reentrenar (~3 s)
"""
import json
import math
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from viz import PALETTE, style_axes

torch.manual_seed(42)
random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

words = open("names.txt", encoding="utf-8").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
V = len(stoi)

random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
SPLITS = {"train": words[:n1], "dev": words[n1:n2], "test": words[n2:]}


def build(ws, block):
    X, Y = [], []
    for w in ws:
        ctx = [0] * block
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def ngram(block, alphas=(1.0, 0.1, 0.01, 0.001)):
    """Tabla de conteos de tamaño |V|^(block+1). Revienta en memoria para block>=6:
    eso mismo ES el resultado — la tabla densa no cabe."""
    D = {k: build(ws, block) for k, ws in SPLITS.items()}

    def flat(X, Y):
        f = torch.zeros(len(Y), dtype=torch.long)
        for i in range(block):
            f = f * V + X[:, i]
        return f * V + Y

    cells = V ** (block + 1)
    if cells * 8 > 8e9:            # torch.bincount devuelve int64: 8 bytes por celda
        return {"oom_bytes": cells * 8, "contextos_posibles": V ** block}

    counts = torch.bincount(flat(*D["train"]), minlength=cells).float().view(-1, V)
    best = None
    for alpha in alphas:
        P = ((counts + alpha) / (counts + alpha).sum(-1, keepdim=True)).view(-1)
        r = {k: -P[flat(*D[k])].log().mean().item() for k in D}
        r["alpha"] = alpha
        if best is None or r["dev"] < best["dev"]:
            best = r
    best["contextos_posibles"] = V ** block
    best["contextos_vistos"] = int((counts.sum(-1) > 0).sum())
    return best


def train_mlp(block=3, n_embd=10, n_hidden=200, steps=30000, seed=42):
    D = {k: build(ws, block) for k, ws in SPLITS.items()}
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((V, n_embd), generator=g)
    H = torch.randn((block * n_embd, n_hidden), generator=g) * 0.1
    d = torch.randn(n_hidden, generator=g) * 0.01
    U = torch.randn((n_hidden, V), generator=g) * 0.01
    b = torch.randn(V, generator=g) * 0
    n_params = sum(p.nelement() for p in [C, H, d, U, b])
    C, H, d, U, b = [p.to(device).requires_grad_() for p in [C, H, d, U, b]]
    ps = [C, H, d, U, b]
    Xtr, Ytr = D["train"][0].to(device), D["train"][1].to(device)

    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (64,), generator=g)
        h = torch.tanh(C[Xtr[ix]].view(64, -1) @ H + d)
        loss = F.cross_entropy(h @ U + b, Ytr[ix])
        for p in ps:
            p.grad = None
        loss.backward()
        lr = 0.1 if step < 20000 else 0.01
        for p in ps:
            p.data += -lr * p.grad

    @torch.no_grad()
    def ev(split):
        X, Y = D[split][0].to(device), D[split][1].to(device)
        h = torch.tanh(C[X].view(X.shape[0], -1) @ H + d)
        return F.cross_entropy(h @ U + b, Y).item()

    return {k: ev(k) for k in D} | {"params": n_params}, C.detach().cpu()


# ---------------------------------------------------------------------------
def exp_contexto(rows=None):
    """El experimento central: ¿a partir de qué contexto gana el modelo neuronal?

    Con `rows` (de una corrida previa) solo re-dibuja, sin reentrenar."""
    print("=== 1. CONTEXTO: dispersión contra generalización ===\n")
    if rows is None:
        rows = []
        print(f"{'ctx':>4} | {'contextos vistos/posibles':>28} | {'N-GRAMA dev':>11} | {'MLP dev':>8}")
        print("-" * 66)
        for block in [1, 2, 3, 4, 5, 6]:
            ng = ngram(block)
            ml, _ = train_mlp(block=block)
            if "oom_bytes" in ng:
                print(f"{block:>4} | {'tabla de ' + str(round(ng['oom_bytes']/1e9)) + ' GB — no cabe':>28} |"
                      f" {'imposible':>11} | {ml['dev']:>8.4f}")
                rows.append({"ctx": block, "ngram": None, "mlp": ml, "oom_gb": ng["oom_bytes"] / 1e9})
                continue
            cov = 100 * ng["contextos_vistos"] / ng["contextos_posibles"]
            print(f"{block:>4} | {ng['contextos_vistos']:>8}/{ng['contextos_posibles']:<9} ({cov:4.1f}%) |"
                  f" {ng['dev']:>11.4f} | {ml['dev']:>8.4f}")
            rows.append({"ctx": block, "ngram": ng, "mlp": ml, "cobertura_pct": cov})
    else:
        print("(re-dibujando desde analysis.json, sin reentrenar)")

    ok = [r for r in rows if r["ngram"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=PALETTE["surface"])
    ax.plot([r["ctx"] for r in ok], [r["ngram"]["dev"] for r in ok],
            color=PALETTE["s2"], lw=2, marker="o", ms=8, zorder=3)
    ax.plot([r["ctx"] for r in rows], [r["mlp"]["dev"] for r in rows],
            color=PALETTE["s1"], lw=2, marker="o", ms=8, zorder=3)

    cross = next((r["ctx"] for r in ok if r["mlp"]["dev"] < r["ngram"]["dev"]), None)
    if cross:
        ax.axvline(cross, color=PALETTE["ink3"], lw=1, ls=":", zorder=1)
        ax.annotate("el MLP toma\nla delantera", xy=(cross, 2.40), xytext=(cross - 0.12, 2.40),
                    fontsize=8.5, color=PALETTE["ink2"], va="center", ha="right")

    # etiquetas directas al final de cada serie
    ax.annotate(f"n-grama de conteo: {ok[-1]['ngram']['dev']:.3f}\n"
                f"solo {ok[-1]['cobertura_pct']:.1f}% de los contextos vistos",
                xy=(ok[-1]["ctx"], ok[-1]["ngram"]["dev"]),
                xytext=(ok[-1]["ctx"] + 0.15, ok[-1]["ngram"]["dev"] - 0.06),
                fontsize=9, color=PALETTE["ink"], va="top")
    ax.annotate(f"MLP: {rows[-1]['mlp']['dev']:.3f}", xy=(rows[-1]["ctx"], rows[-1]["mlp"]["dev"]),
                xytext=(rows[-1]["ctx"] + 0.15, rows[-1]["mlp"]["dev"]),
                fontsize=9, color=PALETTE["ink"], va="center")

    # el punto donde contar deja de ser posible, no solo de ser bueno
    oom = next((r for r in rows if r.get("oom_gb")), None)
    if oom:
        last = ok[-1]
        ax.plot([last["ctx"], oom["ctx"]], [last["ngram"]["dev"], last["ngram"]["dev"] + 0.11],
                color=PALETTE["s2"], lw=1.4, ls=(0, (2, 2)), zorder=2)
        ax.plot([oom["ctx"]], [last["ngram"]["dev"] + 0.11], marker="x", ms=9, mew=2.2,
                color=PALETTE["s2"], zorder=3)
        ax.annotate(f"tabla de {oom['oom_gb']:.0f} GB:\nya no cabe en memoria",
                    xy=(oom["ctx"], last["ngram"]["dev"] + 0.11),
                    xytext=(oom["ctx"] - 0.15, last["ngram"]["dev"] + 0.145),
                    fontsize=8.5, color=PALETTE["s2"], ha="right", va="center")

    ax.set_xlim(0.8, 7.6)
    ax.set_xticks([r["ctx"] for r in rows])
    style_axes(ax, "Al crecer el contexto, contar deja de funcionar",
               "Caracteres de contexto (n-1)", "Cross-entropy loss (dev)")
    fig.subplots_adjust(right=0.8, left=0.09, top=0.86, bottom=0.13)
    fig.savefig("contexto.png", dpi=160, facecolor=PALETTE["surface"])
    print("\n-> contexto.png")
    return rows


def exp_capacidad(rows=None):
    """¿El modelo base está sobre o sub-ajustado?"""
    print("\n=== 2. CAPACIDAD: ¿underfitting u overfitting? ===\n")
    if rows is None:
        rows = []
        print(f"{'m':>3} {'h':>5} | {'params':>7} | {'train':>7} {'dev':>7} {'gap':>7}")
        print("-" * 46)
        for m, h in [(2, 100), (5, 150), (10, 200), (20, 300), (30, 500), (50, 800)]:
            r, _ = train_mlp(n_embd=m, n_hidden=h)
            r["m"], r["n_hidden"] = m, h
            rows.append(r)
            print(f"{m:>3} {h:>5} | {r['params']:>7} | {r['train']:>7.4f} {r['dev']:>7.4f} "
                  f"{r['dev'] - r['train']:>+7.4f}")
    else:
        print("(re-dibujando desde analysis.json, sin reentrenar)")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=PALETTE["surface"])
    xs = [r["params"] for r in rows]
    ax.plot(xs, [r["train"] for r in rows], color=PALETTE["s3"], lw=2, marker="o", ms=7)
    ax.plot(xs, [r["dev"] for r in rows], color=PALETTE["s1"], lw=2, marker="o", ms=7)
    ax.set_xscale("log")
    ax.annotate("train", xy=(xs[-1], rows[-1]["train"]), xytext=(xs[-1] * 1.15, rows[-1]["train"]),
                fontsize=9, color=PALETTE["ink"], va="center")
    ax.annotate("dev", xy=(xs[-1], rows[-1]["dev"]), xytext=(xs[-1] * 1.15, rows[-1]["dev"]),
                fontsize=9, color=PALETTE["ink"], va="center")
    base = next(r for r in rows if r["m"] == 10)
    ax.plot([base["params"]], [base["dev"]], marker="o", ms=13, mfc="none",
            mec=PALETTE["ink2"], mew=1.4)
    ax.annotate("modelo base\n(11,897 params)", xy=(base["params"], base["dev"]),
                xytext=(base["params"], base["dev"] + 0.055), ha="center",
                fontsize=8.5, color=PALETTE["ink2"])

    # el listón a vencer: dónde queda el mejor baseline de conteo
    FOURGRAM_DEV = 2.1018
    ax.axhline(FOURGRAM_DEV, color=PALETTE["s2"], lw=1.6, ls=(0, (5, 3)), zorder=1)
    ax.annotate(f"baseline 4-grama  {FOURGRAM_DEV:.3f}", xy=(xs[0], FOURGRAM_DEV),
                xytext=(xs[0] * 1.05, FOURGRAM_DEV + 0.008), fontsize=8.5,
                color=PALETTE["s2"], va="bottom")

    style_axes(ax, "El dev loss sigue bajando: el modelo base está underfitting",
               "Parámetros (escala log)", "Cross-entropy loss")
    fig.subplots_adjust(right=0.86, left=0.09, top=0.86, bottom=0.13)
    fig.savefig("capacidad.png", dpi=160, facecolor=PALETTE["surface"])
    print("\n-> capacidad.png")
    return rows


def exp_embeddings(cached=None):
    """El mecanismo del paper, hecho imagen."""
    print("\n=== 3. EMBEDDINGS: ¿qué aprendió la matriz C? ===\n")
    if cached is None:
        r, C = train_mlp(n_embd=2, n_hidden=200)
        print(f"m=2 -> train {r['train']:.4f}  dev {r['dev']:.4f}")
    else:
        print("(re-dibujando desde analysis.json, sin reentrenar)")
        r = cached["loss"]
        C = torch.tensor([cached["coords"][itos[i]] for i in range(V)])

    vowels = set("aeiou")
    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=PALETTE["surface"])
    for i in range(V):
        ch = itos[i]
        if ch == ".":
            col, label = PALETTE["ink2"], "·"
        elif ch in vowels:
            col, label = PALETTE["s2"], ch
        else:
            col, label = PALETTE["s1"], ch
        ax.scatter(C[i, 0], C[i, 1], s=320, color=col, alpha=0.16, zorder=2)
        ax.text(C[i, 0], C[i, 1], label, ha="center", va="center",
                fontsize=13, color=col, zorder=3, fontweight="bold")

    # la 'y' es el caso interesante: semivocal, y el modelo la coloca con las vocales
    yx, yy = float(C[stoi["y"], 0]), float(C[stoi["y"], 1])
    ax.annotate("la 'y' cae del lado de las vocales\n(nadie se lo dijo: es semivocal)",
                xy=(yx, yy), xytext=(yx + 0.45, yy - 0.42), fontsize=8.5,
                color=PALETTE["ink2"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["ink3"], lw=0.8,
                                shrinkA=2, shrinkB=10))

    ax.scatter([], [], s=90, color=PALETTE["s2"], alpha=0.6, label="vocales")
    ax.scatter([], [], s=90, color=PALETTE["s1"], alpha=0.6, label="consonantes")
    ax.scatter([], [], s=90, color=PALETTE["ink2"], alpha=0.6, label="token inicio/fin")
    ax.legend(loc="center right", frameon=False, fontsize=9, labelcolor=PALETTE["ink2"])
    style_axes(ax, "La tabla C aprende que las vocales se comportan parecido",
               "dimensión 1", "dimensión 2")
    ax.grid(True, color=PALETTE["ink3"], alpha=0.14, lw=0.7)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.09)
    fig.savefig("embeddings.png", dpi=160, facecolor=PALETTE["surface"])
    print("-> embeddings.png")
    return {"loss": r, "coords": {itos[i]: [round(float(C[i, 0]), 3), round(float(C[i, 1]), 3)]
                                  for i in range(V)}}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "todos"

    # `replot` re-dibuja las figuras desde analysis.json sin reentrenar nada
    cache = {}
    if which == "replot":
        cache = json.load(open("analysis.json", encoding="utf-8"))
        which = "todos"

    out = {}
    if which in ("todos", "contexto"):
        out["contexto"] = exp_contexto(cache.get("contexto"))
    if which in ("todos", "capacidad"):
        out["capacidad"] = exp_capacidad(cache.get("capacidad"))
    if which in ("todos", "embeddings"):
        out["embeddings"] = exp_embeddings(cache.get("embeddings"))

    with open("analysis.json", "w", encoding="utf-8") as f:
        json.dump(out | {k: v for k, v in cache.items() if k not in out}, f,
                  indent=2, ensure_ascii=False)
    print("\nGuardado analysis.json")
