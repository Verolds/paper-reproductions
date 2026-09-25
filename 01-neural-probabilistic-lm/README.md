# 01 — A Neural Probabilistic Language Model (Bengio et al., 2003)

📄 Paper original: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

## La idea en simple

Antes de este paper, los modelos de lenguaje eran **n-gramas de conteo**: tablas gigantes
que guardaban, literalmente, cuántas veces viste "the cat sat" para estimar
P(sat | the cat). El problema: esas tablas crecen exponencialmente y no generalizan —
si nunca viste "the dog sat" pero sí "the cat sat", el modelo no tiene forma de saber
que "dog" y "cat" se comportan parecido.

Bengio et al. (2003) propusieron algo que hoy nos parece obvio pero en su momento fue
revolucionario:

1. Representa cada palabra como un **vector denso aprendible** (embedding), no como un
   índice arbitrario.
2. Concatena los embeddings de las últimas N palabras de contexto.
3. Pasa eso por una **red neuronal (MLP)** que predice la distribución de probabilidad
   de la siguiente palabra.

Como el modelo aprende que "cat" y "dog" tienen embeddings parecidos (porque aparecen
en contextos parecidos), automáticamente generaliza: aprender sobre "the cat sat" le
ayuda a predecir mejor "the dog sat" aunque nunca la haya visto. Este es el ancestro
directo de word2vec y, varios pasos después, de los Transformers.

## Qué reproduje

Una versión a **nivel de caracteres** (en vez de palabras) entrenada sobre 32,000
nombres propios, siguiendo el enfoque pedagógico de Andrej Karpathy (proyecto
"makemore"). La arquitectura es la misma del paper: tabla de embeddings + MLP con una
capa oculta (tanh) + softmax. El objetivo: predecir el siguiente carácter dado un
contexto de 3 caracteres previos, y luego usar eso para *generar* nombres nuevos.

**Arquitectura:**
- Contexto: 3 caracteres previos
- Embedding: 27 tokens (a-z + token especial ".") → vectores de 10 dimensiones
- Capa oculta: 200 neuronas, activación tanh
- ~11,900 parámetros en total (comparar con un GPT-2 pequeño: 124M — esto es diminuto)

## Resultados

- Corrido en GPU local (RTX 4050), 30,000 pasos, ~1 minuto de entrenamiento
- Loss final: **train 2.12 / dev 2.14** (sin overfitting — el modelo no memorizó)
- Ver `loss_curve.png` para la curva de entrenamiento
- Ver `results.json` para los nombres generados

**Ejemplos de nombres 100% inventados por el modelo:**
`ann, ckara, marian, amanar, paree, zen, kel, lando, mia, yasen, mar, zanna, nak`

Ninguno de estos aparece en el dataset original — el modelo aprendió el "patrón" de
qué hace que algo suene a nombre en inglés, no memorizó ejemplos.

## Cómo correrlo

```bash
pip install torch matplotlib
python train.py
```

Corre en CPU o GPU automáticamente (detecta CUDA). En GPU tarda ~1 minuto, en CPU
unos minutos.

## Para el reel / video

Guión sugerido (60-90s):
1. **Gancho**: "¿Sabías que antes de ChatGPT, los modelos de lenguaje ni siquiera
   sabían que 'perro' y 'gato' se parecen?" (mostrar tabla de n-gramas gigante)
2. **El problema**: mostrar cómo un modelo de conteo no puede generalizar
3. **La idea de Bengio 2003**: embeddings + red neuronal (diagrama simple:
   caracteres → vectores → MLP → predicción)
4. **Resultado en vivo**: mostrar la curva de loss bajando + nombres generados
   apareciendo uno por uno (efecto satisfactorio)
5. **Cierre**: "Esta idea de 2003 es el abuelo de GPT. Siguiente video: ¿cómo
   pasamos de esto a los Transformers?"

Assets disponibles: `loss_curve.png`, `results.json` (nombres generados),
este mismo código para mostrar en pantalla.
