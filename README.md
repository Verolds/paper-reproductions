# Paper Reproductions

Recreando papers clave de investigación en IA, desde cero, en orden creciente de complejidad — construyendo intuición paso a paso desde embeddings simples hasta Transformers y fine-tuning eficiente (LoRA).

Cada carpeta = un paper reproducido con: implementación en PyTorch, resultados/gráficas, y un resumen explicado en simple para acompañar el contenido en video/reels.

## Roadmap

| # | Paper | Año | Idea central | Estado |
|---|-------|-----|---------------|--------|
| 01 | [A Neural Probabilistic Language Model](./01-neural-probabilistic-lm) (Bengio et al.) | 2003 | Primer modelo de lenguaje neuronal: embeddings + MLP | ✅ completo |
| 02 | word2vec (Mikolov et al.) | 2013 | Embeddings de palabras vía skip-gram | ⬜ pendiente |
| 03 | Attention (Bahdanau et al.) | 2014 | Atención en seq2seq con RNNs | ⬜ pendiente |
| 04 | Attention Is All You Need (Vaswani et al.) | 2017 | El Transformer — mini-GPT char-level | ⬜ pendiente |
| 05 | LoRA (Hu et al.) | 2021 | Fine-tuning eficiente de LLMs | ⬜ pendiente |

## Resultado destacado (01)

El MLP de Bengio **pierde** contra un 4-grama de conteo a |V|=27 — y ese resultado
negativo es el hallazgo: la maldición de la dimensionalidad que el paper resuelve no
existe a esa escala. Creciendo el contexto hasta donde la tabla de conteo pediría 84 GB,
la ventaja se invierte. [Ver el análisis completo →](./01-neural-probabilistic-lm)

## Setup

- GPU local: NVIDIA RTX 4050 Laptop (6GB VRAM), CUDA vía PyTorch 2.3.1+cu121
- Fallback: Google Colab / Kaggle (notebooks incluidos, `.ipynb` corren igual ahí)

## Autor

Diego — estudiante de IA en Ciudad de México.
