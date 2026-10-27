import numpy as np


class ModeloFalso:
    """Embedding de brinquedo (contagem de letras): o teste não baixa modelo nenhum."""

    def encode(self, textos, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
               convert_to_numpy=True):
        vetores = np.zeros((len(textos), 26), dtype="float32")
        for i, texto in enumerate(textos):
            for letra in texto.lower():
                if "a" <= letra <= "z":
                    vetores[i, ord(letra) - ord("a")] += 1
        return vetores / np.linalg.norm(vetores, axis=1, keepdims=True)
