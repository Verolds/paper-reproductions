"""
Reproducción de: "A Neural Probabilistic Language Model" (Bengio, Ducharme, Vincent, Jauvin, 2003)
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

La afirmación del paper no es "los embeddings + MLP funcionan". Es que SUPERAN a los
n-gramas de conteo suavizados (~24% mejor perplejidad, Tabla 1). Por eso este script
entrena los dos y los compara sobre los mismos splits: sin el baseline no hay
reproducción, solo una implementación de la arquitectura.

Nivel de CARACTERES sobre 32,033 nombres propios (enfoque pedagógico de Karpathy,
"makemore"): misma arquitectura del paper, entrena en ~1 minuto en una GPU de laptop.

    python train.py            # corrida canónica -> loss_curve.png, results.json
    python analysis.py         # los tres experimentos -> contexto.png, capacidad.png, embeddings.png
"""
import json
import random

import matplotlib
matplotlib.use("Agg")            # script sin pantalla: escribe PNG. En el notebook va %matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from viz import PALETTE, style_axes

torch.manual_seed(42)
random.seed(42)

# --- hiperparámetros ---------------------------------------------------------
BLOCK_SIZE = 3        # n-1 en el paper: caracteres de contexto
N_EMBD = 10           # m en el paper: dimensión de cada embedding
N_HIDDEN = 200        # h en el paper: neuronas de la capa oculta
MAX_STEPS = 30000
BATCH_SIZE = 64
MAX_NAME_LEN = 32     # cota de seguridad al generar

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
words = open("names.txt", encoding="utf-8").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0                                  # token especial de inicio/fin
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)


def build_dataset(ws, block_size=BLOCK_SIZE):
    """Ventana deslizante: cada fila es (contexto de block_size chars) -> (char siguiente)."""
    X, Y = [], []
    for w in ws:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
words_tr, words_dev, words_te = words[:n1], words[n1:n2], words[n2:]

# los tres splits se mueven a device juntos — si dejas test en CPU, evaluarlo truena
Xtr, Ytr = (t.to(device) for t in build_dataset(words_tr))
Xdev, Ydev = (t.to(device) for t in build_dataset(words_dev))
Xte, Yte = (t.to(device) for t in build_dataset(words_te))

print(f"Vocabulario |V| = {vocab_size}")
print(f"train={Xtr.shape[0]}  dev={Xdev.shape[0]}  test={Xte.shape[0]}")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# 2. Baseline: n-gramas de conteo (contra esto compite el paper)
# ---------------------------------------------------------------------------
def ngram_baseline(order, alphas=(1.0, 0.1, 0.01, 0.001)):
    """Conteos + suavizado add-alpha. `order` = caracteres de contexto.
    Elige alpha por dev, y reporta los tres splits."""
    def flat(X, Y):
        """(contexto, objetivo) -> un solo índice en la tabla aplanada."""
        f = torch.zeros(len(Y), dtype=torch.long, device=X.device)
        for i in range(order):
            f = f * vocab_size + X[:, BLOCK_SIZE - order + i]
        return f * vocab_size + Y

    counts = torch.bincount(
        flat(Xtr, Ytr), minlength=vocab_size ** (order + 1)
    ).float().view(-1, vocab_size)

    best = None
    for alpha in alphas:
        P = ((counts + alpha) / (counts + alpha).sum(-1, keepdim=True)).view(-1)
        r = {split: -P[flat(X, Y)].log().mean().item()
             for split, (X, Y) in [("train", (Xtr, Ytr)), ("dev", (Xdev, Ydev)),
                                   ("test", (Xte, Yte))]}
        r["alpha"] = alpha
        if best is None or r["dev"] < best["dev"]:
            best = r
    return best


print("\n--- Baselines de conteo ---")
baselines = {}
for order, name in [(1, "bigrama"), (2, "trigrama"), (3, "4-grama")]:
    baselines[name] = ngram_baseline(order)
    r = baselines[name]
    print(f"{name:>9} ({order} char ctx): train {r['train']:.4f}  dev {r['dev']:.4f}  "
          f"test {r['test']:.4f}   [alpha={r['alpha']}]")

best_ngram_name = min(baselines, key=lambda k: baselines[k]["dev"])
best_ngram = baselines[best_ngram_name]

# ---------------------------------------------------------------------------
# 3. El modelo del paper:  y = b + U·tanh(d + Hx),  x = concat(C[w_{t-1}], ...)
# ---------------------------------------------------------------------------
g = torch.Generator().manual_seed(42)

# Se crean en CPU y se multiplican por su escalar ANTES de .to(device).requires_grad_():
# así quedan como leaf tensors de verdad. Al revés, son el resultado de una operación
# y el optimizador no puede actualizarlos in-place.
C = torch.randn((vocab_size, N_EMBD), generator=g)                      # |V| x m
H = torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN), generator=g) * 0.1     # (n-1)m x h
d = torch.randn(N_HIDDEN, generator=g) * 0.01                           # h
U = torch.randn((N_HIDDEN, vocab_size), generator=g) * 0.01             # h x |V|
b = torch.randn(vocab_size, generator=g) * 0                            # |V|

parameters = [C, H, d, U, b]
n_params = sum(p.nelement() for p in parameters)
parameters = [p.to(device).requires_grad_() for p in parameters]
C, H, d, U, b = parameters
print(f"\nParámetros del MLP: {n_params}")


def forward(X):
    emb = C[X]                                        # lookup   -> (batch, block, m)
    x = emb.view(emb.shape[0], -1)                    # concatena -> (batch, block*m)
    h = torch.tanh(x @ H + d)                         # capa oculta
    return h @ U + b                                  # logits


# ---------------------------------------------------------------------------
# 4. Entrenamiento
# ---------------------------------------------------------------------------
lossi = []
for step in range(MAX_STEPS):
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    loss = F.cross_entropy(forward(Xb), Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if step < 20000 else 0.01        # a un décimo después de 20k pasos
    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.item())
    if step % 5000 == 0:
        print(f"step {step:6d}/{MAX_STEPS}  loss {loss.item():.4f}")


@torch.no_grad()
def split_loss(X, Y):
    return F.cross_entropy(forward(X), Y).item()


mlp = {"train": split_loss(Xtr, Ytr), "dev": split_loss(Xdev, Ydev),
       "test": split_loss(Xte, Yte)}
print(f"\nMLP    -> train {mlp['train']:.4f}  dev {mlp['dev']:.4f}  test {mlp['test']:.4f}")
print(f"{best_ngram_name:>6} -> train {best_ngram['train']:.4f}  dev {best_ngram['dev']:.4f}  "
      f"test {best_ngram['test']:.4f}")

delta = (1 - mlp["dev"] / best_ngram["dev"]) * 100
veredicto = "el MLP GANA" if delta > 0 else "el MLP PIERDE"
print(f"\n{veredicto} por {abs(delta):.1f}% en dev contra el mejor n-grama ({best_ngram_name}).")
print("Con |V|=27 y 3 chars de contexto los conteos todavía son densos (27.6% de los")
print("contextos vistos en train), así que no hay dispersión que castigar. Corre")
print("analysis.py para ver dónde se invierte el resultado.")

# ---------------------------------------------------------------------------
# 5. Generación
# ---------------------------------------------------------------------------
g_dev = torch.Generator(device=device).manual_seed(42)   # mismo device que el modelo


@torch.no_grad()
def generate(n=20):
    names = []
    for _ in range(n):
        out, context = [], [0] * BLOCK_SIZE
        while len(out) < MAX_NAME_LEN:
            probs = F.softmax(forward(torch.tensor([context], device=device)), dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g_dev).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


generated = generate(20)
existing = set(words)
novel = [n for n in generated if n not in existing]
print(f"\nNombres generados ({len(novel)}/{len(generated)} no están en el dataset):")
for name in generated:
    print(f"  {name:<16}{'' if name in existing else '(nuevo)'}")

# ---------------------------------------------------------------------------
# 6. Curva de entrenamiento
# ---------------------------------------------------------------------------
# El loss por batch (B=64) tiene un rango de ~±0.6; la señal que interesa es de ~0.06.
# Sin suavizar, el ruido es ~18x la señal y la curva es una mancha ilegible.
W = 250
smooth = [sum(lossi[max(0, i - W):i + 1]) / len(lossi[max(0, i - W):i + 1])
          for i in range(len(lossi))]

fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=PALETTE["surface"])
ax.plot(lossi, color=PALETTE["band"], lw=0.35, alpha=0.55, zorder=1)
for y, col in [(baselines["trigrama"]["dev"], PALETTE["s3"]),
               (baselines["4-grama"]["dev"], PALETTE["s2"])]:
    ax.axhline(y, color=col, lw=1.6, ls=(0, (5, 3)), zorder=2)
ax.plot(smooth, color=PALETTE["s1"], lw=2, zorder=4)
ax.axvline(20000, color=PALETTE["ink3"], lw=1, ls=":", zorder=3)
ax.annotate("learning rate 0.1 → 0.01", xy=(20000, 2.62), xytext=(19200, 2.62),
            ha="right", va="center", fontsize=8.5, color=PALETTE["ink2"])

# etiquetas directas, separadas si se encimarían
labels = sorted([(smooth[-1], f"MLP  {smooth[-1]:.3f}", PALETTE["s1"]),
                 (baselines["4-grama"]["dev"], f"baseline 4-grama  {baselines['4-grama']['dev']:.3f}", PALETTE["s2"]),
                 (baselines["trigrama"]["dev"], f"baseline trigrama  {baselines['trigrama']['dev']:.3f}", PALETTE["s3"])])
placed = []
for y, txt, col in labels:
    ly = y if not placed else max(y, placed[-1] + 0.075)
    placed.append(ly)
    ax.annotate(txt, xy=(MAX_STEPS, y), xytext=(MAX_STEPS + 500, ly), va="center",
                fontsize=9, color=PALETTE["ink"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["ink3"], lw=0.7,
                                shrinkA=0, shrinkB=2) if abs(ly - y) > 1e-9 else None)
    ax.plot([MAX_STEPS + 200], [y], marker="o", ms=5, color=col, clip_on=False, zorder=5)

ax.set_xlim(0, MAX_STEPS)
ax.set_ylim(1.5, 2.8)
style_axes(ax, "Curva de entrenamiento — MLP contra los baselines de conteo",
           "Paso de entrenamiento", "Cross-entropy loss")
fig.subplots_adjust(right=0.74, left=0.08, top=0.86, bottom=0.13)
fig.savefig("loss_curve.png", dpi=160, facecolor=PALETTE["surface"])
print("\nGuardado loss_curve.png")

# ---------------------------------------------------------------------------
# 7. Resultados
# ---------------------------------------------------------------------------
import math

results = {
    "paper": "Bengio et al. 2003 - A Neural Probabilistic Language Model",
    "reproduccion": "nivel de caracteres, 32,033 nombres propios (dataset karpathy/makemore)",
    "arquitectura": {
        "vocab_size": vocab_size,
        "block_size (n-1)": BLOCK_SIZE,
        "n_embd (m)": N_EMBD,
        "n_hidden (h)": N_HIDDEN,
        "n_params": n_params,
    },
    "mlp": {k: round(v, 4) for k, v in mlp.items()}
           | {f"perplexity_{k}": round(math.exp(v), 2) for k, v in mlp.items()},
    "baselines_ngrama": {
        name: {k: round(v, 4) for k, v in r.items() if k != "alpha"} | {"alpha": r["alpha"]}
        for name, r in baselines.items()
    },
    "veredicto": {
        "mejor_ngrama": best_ngram_name,
        "delta_dev_pct": round(delta, 2),
        "resumen": f"{veredicto} por {abs(delta):.1f}% contra el {best_ngram_name}",
    },
    "nombres_generados": generated,
    "nombres_no_en_dataset": novel,
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Guardado results.json")
