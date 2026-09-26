from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Sequence


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434",
)

EMBEDDING_MODEL = os.getenv(
    "ZOE_EMBEDDING_MODEL",
    "qwen3-embedding:0.6b",
)

EMBED_TIMEOUT = float(
    os.getenv(
        "ZOE_EMBED_TIMEOUT",
        "120",
    )
)


# ============================================================
# PROVIDER
# ============================================================

class EmbeddingProvider:

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_URL,
    ) -> None:

        self.model = str(model)

        self.url = (
            str(base_url).rstrip("/")
            + "/api/embed"
        )

    # ========================================================
    # SINGLE
    # ========================================================

    def embed(
        self,
        text: str,
    ) -> list[float]:

        text = str(text).strip()

        if not text:
            raise ValueError(
                "Cannot embed empty text."
            )

        results = self.embed_batch(
            [text]
        )

        if len(results) != 1:
            raise RuntimeError(
                "Embedding model returned "
                "an unexpected number of embeddings."
            )

        return results[0]

    # ========================================================
    # BATCH
    # ========================================================

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        clean = [
            str(text).strip()
            for text in texts
        ]

        if not clean:
            return []

        if any(
            not text
            for text in clean
        ):
            raise ValueError(
                "Embedding input contains empty text."
            )

        payload = json.dumps(
            {
                "model": self.model,
                "input": clean,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=EMBED_TIMEOUT,
            ) as response:

                raw = response.read()

        except urllib.error.URLError as exc:

            raise RuntimeError(
                "Could not connect to Ollama "
                f"at {self.url}: {exc}"
            ) from exc

        try:

            data = json.loads(
                raw.decode("utf-8")
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from exc

        embeddings = data.get(
            "embeddings"
        )

        if not isinstance(
            embeddings,
            list,
        ):

            raise RuntimeError(
                "Ollama response did not contain "
                "'embeddings'."
            )

        if len(embeddings) != len(clean):

            raise RuntimeError(
                "Ollama returned "
                f"{len(embeddings)} embeddings "
                f"for {len(clean)} inputs."
            )

        result: list[list[float]] = []

        for embedding in embeddings:

            if not isinstance(
                embedding,
                list,
            ):

                raise RuntimeError(
                    "Ollama returned an invalid embedding."
                )

            values = [
                float(value)
                for value in embedding
            ]

            if not values:
                raise RuntimeError(
                    "Ollama returned an empty embedding."
                )

            result.append(values)

        return result


# ============================================================
# GLOBAL
# ============================================================

embedding_provider = EmbeddingProvider()