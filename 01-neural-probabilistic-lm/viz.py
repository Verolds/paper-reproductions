"""Paleta y estilo compartidos por train.py y analysis.py.

Los tres tonos de serie están validados para daltonismo (peor par adyacente
ΔE 9.2 en deuteranopía, sobre un objetivo de ≥8) y todas las series van con
etiqueta directa, nunca identificadas solo por color.
"""

PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",      # texto primario
    "ink2": "#52514e",     # texto secundario: ejes, etiquetas
    "ink3": "#8a8983",     # texto terciario: grid, spines
    "s1": "#2a78d6",       # azul    — el modelo neuronal
    "s2": "#eb6834",       # naranja — n-gramas de conteo
    "s3": "#1baf7a",       # aqua    — tercera serie
    "band": "#d8d7d2",     # ruido de fondo, recesivo
}


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Ejes recesivos: sin marco superior/derecho, grid tenue detrás de los datos."""
    ax.set_facecolor(PALETTE["surface"])
    if title:
        ax.set_title(title, fontsize=12, color=PALETTE["ink"], loc="left", pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=PALETTE["ink2"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=PALETTE["ink2"])
    ax.grid(axis="y", color=PALETTE["ink3"], alpha=0.18, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["ink3"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=PALETTE["ink2"], labelsize=8.5, length=3)
    return ax
