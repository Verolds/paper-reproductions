"""
Reproducción de: "A Neural Probabilistic Language Model" (Bengio, Ducharme, Vincent, Jauvin, 2003)
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

Idea central del paper:
    Antes de esto, los modelos de lenguaje eran n-gramas de conteo (tablas gigantes de
    probabilidades P(palabra | contexto)). Bengio et al. propusieron en su lugar:
      1. Representar cada palabra/token como un VECTOR denso aprendible (embedding).
      2. Concatenar los embeddings del contexto (las últimas N palabras).
      3. Pasar eso por un MLP (capa oculta + tanh) que predice la distribución de
         probabilidad sobre la siguiente palabra/token.
    Esto resolvió la "maldición de la dimensionalidad" de los n-gramas y es el ancestro
    directo de word2vec, y en última instancia de los Transformers modernos.

Esta reproducción es a nivel de CARACTERES (no palabras) sobre un dataset de nombres,
siguiendo el enfoque pedagógico de Andrej Karpathy (makemore) porque entrena en segundos
y es fácil de visualizar. La arquitectura es exactamente la del paper: embedding table +
MLP con una capa oculta.
"""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import json

torch.manual_seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# 1. Datos: leemos los nombres y construimos el vocabulario de caracteres
# ---------------------------------------------------------------------------
words = open("names.txt", "r", encoding="utf-8").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0  # token especial de inicio/fin de palabra
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)
print(f"Vocabulario ({vocab_size} tokens): {itos}")

BLOCK_SIZE = 3  # tamaño de contexto: cuántos caracteres previos usamos para predecir el siguiente


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])       # 80% train
Xdev, Ydev = build_dataset(words[n1:n2])   # 10% dev
Xte, Yte = build_dataset(words[n2:])       # 10% test
print(f"train={Xtr.shape[0]}  dev={Xdev.shape[0]}  test={Xte.shape[0]}")

# ---------------------------------------------------------------------------
# 2. El modelo: exactamente la arquitectura del paper de Bengio 2003
#    tabla de embeddings -> concatenar contexto -> capa oculta (tanh) -> softmax
# ---------------------------------------------------------------------------
N_EMBD = 10       # dimensión de cada embedding de caracter
N_HIDDEN = 200     # neuronas en la capa oculta del MLP

g = torch.Generator().manual_seed(42)
# NOTA: multiplicamos por el escalar ANTES de marcar requires_grad=True, y solo
# entonces movemos a GPU y activamos requires_grad — así quedan como leaf tensors
# "de verdad" (si no, W1/b1/etc terminan siendo resultado de una operación y
# PyTorch no puede optimizarlas directamente ni moverlas de device limpiamente).
C = torch.randn((vocab_size, N_EMBD), generator=g)                    # tabla de embeddings
W1 = torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN), generator=g) * 0.1
b1 = torch.randn(N_HIDDEN, generator=g) * 0.01
W2 = torch.randn((N_HIDDEN, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

parameters = [C, W1, b1, W2, b2]
n_params = sum(p.nelement() for p in parameters)
print(f"Número de parámetros: {n_params}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando device: {device}")
parameters = [p.to(device).requires_grad_() for p in parameters]
C, W1, b1, W2, b2 = parameters
Xtr, Ytr, Xdev, Ydev = Xtr.to(device), Ytr.to(device), Xdev.to(device), Ydev.to(device)

# ---------------------------------------------------------------------------
# 3. Entrenamiento
# ---------------------------------------------------------------------------
MAX_STEPS = 30000
BATCH_SIZE = 64
lossi = []

for step in range(MAX_STEPS):
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]                              # (batch, block_size, n_embd)
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)   # capa oculta
    logits = h @ W2 + b2                     # capa de salida
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if step < 20000 else 0.01       # decay simple del learning rate
    for p in parameters:
        p.data += -lr * p.grad

    if step % 2000 == 0:
        print(f"step {step:6d}/{MAX_STEPS}  loss {loss.item():.4f}")
    lossi.append(loss.log10().item())

# ---------------------------------------------------------------------------
# 4. Evaluación
# ---------------------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y):
    emb = C[X]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


train_loss = split_loss(Xtr, Ytr)
dev_loss = split_loss(Xdev, Ydev)
print(f"\nLoss final -> train: {train_loss:.4f}  dev: {dev_loss:.4f}")

# ---------------------------------------------------------------------------
# 5. Generación: muestreamos nombres nuevos del modelo entrenado
# ---------------------------------------------------------------------------
g2 = torch.Generator(device=device).manual_seed(2026)
generated = []
for _ in range(20):
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        emb = C[torch.tensor([context], device=device)]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g2).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    generated.append("".join(out))

print("\nNombres generados por el modelo (nunca vistos en el dataset):")
for name in generated:
    print(" -", name)

# ---------------------------------------------------------------------------
# 6. Guardar resultados para el README / reel
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(lossi)
plt.xlabel("step")
plt.ylabel("log10(loss)")
plt.title("Bengio 2003 MLP LM - curva de entrenamiento")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("\nGuardado loss_curve.png")

results = {
    "vocab_size": vocab_size,
    "n_params": n_params,
    "train_loss": train_loss,
    "dev_loss": dev_loss,
    "generated_names": generated,
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Guardado results.json")
