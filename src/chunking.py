from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text.strip():
            return []
        
        delimiters = r'([.!?]\s+|\.\n)'
        parts = re.split(delimiters, text)

        sentences = []
        current_sentence = ""

        for part in parts:
            current_sentence += part
            if part in {'. ', '! ', '? ', '.\n'}:
                cleaned_sentence = current_sentence.strip()
                if cleaned_sentence:
                    sentences.append(cleaned_sentence)
                current_sentence = ""

        final_sentence = current_sentence.strip()
        if final_sentence:
            sentences.append(final_sentence)

        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences).strip()
            chunks.append(chunk_text)
        
        return chunks
        raise NotImplementedError("Implement SentenceChunker.chunk")


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        if not text.strip():
            return []
        return self._split(text, self.separators)
        raise NotImplementedError("Implement RecursiveChunker.chunk")

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        separator = ""
        next_separators = []

        for i, sep in enumerate(remaining_separators):
            if sep == "":
                separator = sep
                break
            if sep in current_text:
                separator = sep
                next_separators = remaining_separators[i + 1:]
                break
        if separator == "":
            return [current_text[i:i + self.chunk_size] 
                    for i in range(0, len(current_text), self.chunk_size)]
        
        splits = current_text.split(separator)

        final_chunks = []
        current_chunk_pieces = []
        current_length = 0

        for piece in splits:
            if len(piece) > self.chunk_size:
                if current_chunk_pieces:
                    final_chunks.append(separator.join(current_chunk_pieces))
                    current_chunk_pieces = []
                    current_length = 0
                recursive_chunks = self._split(piece, next_separators)
                final_chunks.extend(recursive_chunks)
                continue

            separator_len = len(separator) if current_length > 0 else 0
            new_length = current_length + separator_len + len(piece)

            if new_length > self.chunk_size:
                if current_chunk_pieces:
                    final_chunks.append(separator.join(current_chunk_pieces))
                current_chunk_pieces = [piece]
                current_length = len(piece)
            else:
                current_chunk_pieces.append(piece)
                current_length += separator_len + len(piece)
        
        if current_chunk_pieces:
            final_chunks.append(separator.join(current_chunk_pieces))
        
        return final_chunks

        raise NotImplementedError("Implement RecursiveChunker._split")


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
    
    mag_a = math.sqrt(sum(x ** 2 for x in vec_a))
    mag_b = math.sqrt(sum(y ** 2 for y in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)

    raise NotImplementedError("Implement compute_similarity")


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        if not text or not text.strip():
            return {}
        
        sentences_per_chunk = max(1, chunk_size // 100)

        sentence_chunker = SentenceChunker(max_sentences_per_chunk=sentences_per_chunk)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        naive_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        return {
            "Naive_FixedSize": self._compute_stats(naive_chunks),
            "SentenceChunker": self._compute_stats(sentence_chunks),
            "RecursiveChunker": self._compute_stats(recursive_chunks)
        }
    
    def _compute_stats(self, chunks: list[str]) -> dict:
        """
        Helper method to calculate statistics for a list of chunks.
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "avg_length": 0,
                "max_length": 0,
                "min_length": 0,
                "chunks": [] 
            }
        
        lengths = [len(c) for c in chunks]

        return {
            "num_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "chunks": chunks
        }

        raise NotImplementedError("Implement ChunkingStrategyComparator.compare")
