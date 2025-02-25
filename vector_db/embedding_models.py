from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding

OPENAI_EMBEDDING_MODEL = "openai_ada"
OPENAI_EMBEDDING_DIMENSION = 1536

LATE_INTERACTION_MODEL = "colbertv2.0"
LATE_INTERACTION_DIMENSION = 128

LEXICAL_MODEL = "bm25"

BM25_EMBEDDING_MODEL = Bm25("Qdrant/bm25")
LATE_INTERACTION_EMBEDDING_MODEL = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
