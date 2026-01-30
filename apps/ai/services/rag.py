"""
=============================================================================
SPS Assistant - RAG Service
=============================================================================
Retrieval-Augmented Generation for SPS documentation.

LAYMAN EXPLANATION:
RAG is like having an open-book exam. Instead of relying only on
what the AI "remembers" from training, we first search our documents
for relevant information, then include that in the prompt so the AI
can give more accurate, grounded answers.
=============================================================================
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from ..models import Document, DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates text embeddings for semantic search.
    
    Embeddings are vector representations of text that capture
    semantic meaning, allowing us to find similar content.
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                try:
                    import openai
                    self._client = openai.OpenAI(api_key=api_key)
                except ImportError:
                    logger.warning("OpenAI not installed, using mock embeddings")
        return self._client
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.client:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Embedding error: {e}")
        
        # Fallback: simple hash-based mock embedding
        return self._mock_embedding(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.client:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
        
        # Fallback
        return [self._mock_embedding(text) for text in texts]
    
    def _mock_embedding(self, text: str, dim: int = 256) -> List[float]:
        """Generate a simple mock embedding based on text hash."""
        # Use hash to generate deterministic but simple embedding
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(dim).tolist()


class RAGService:
    """
    Retrieval-Augmented Generation service.
    
    Handles:
    1. Document indexing (chunking + embedding)
    2. Semantic search (finding relevant chunks)
    3. Context building for LLM prompts
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chunk_size = 512  # characters
        self.chunk_overlap = 50
        self.top_k = 5
    
    def index_document(self, document: Document) -> int:
        """
        Index a document for RAG retrieval.
        
        Process:
        1. Extract text from file
        2. Split into overlapping chunks
        3. Generate embeddings for each chunk
        4. Store in database
        
        Args:
            document: Document model instance
            
        Returns:
            Number of chunks created
        """
        document.status = Document.Status.INDEXING
        document.save()
        
        try:
            # Extract text
            text = self._extract_text(document)
            
            # Chunk text
            chunks = self._chunk_text(text)
            
            # Delete existing chunks
            DocumentChunk.objects.filter(document=document).delete()
            
            # Generate embeddings in batches
            batch_size = 50
            chunk_objects = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                embeddings = self.embedding_service.embed_batch([c['content'] for c in batch])
                
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    chunk_objects.append(DocumentChunk(
                        document=document,
                        content=chunk['content'],
                        chunk_index=i + j,
                        start_char=chunk['start'],
                        end_char=chunk['end'],
                        embedding=embedding,
                    ))
            
            # Bulk create
            DocumentChunk.objects.bulk_create(chunk_objects)
            
            # Update document
            document.status = Document.Status.INDEXED
            document.chunk_count = len(chunk_objects)
            document.save()
            
            logger.info(f"Indexed document {document.name}: {len(chunk_objects)} chunks")
            return len(chunk_objects)
            
        except Exception as e:
            logger.error(f"Indexing error for {document.name}: {e}")
            document.status = Document.Status.FAILED
            document.error_message = str(e)
            document.save()
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_types: Filter by document type
            
        Returns:
            List of relevant chunks with metadata
        """
        top_k = top_k or self.top_k
        
        # Get query embedding
        query_embedding = self.embedding_service.embed(query)
        
        # Get all indexed chunks
        chunks_qs = DocumentChunk.objects.filter(
            document__status=Document.Status.INDEXED
        ).select_related('document')
        
        if document_types:
            chunks_qs = chunks_qs.filter(document__document_type__in=document_types)
        
        # Calculate similarities
        results = []
        for chunk in chunks_qs:
            if chunk.embedding:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                results.append({
                    'chunk_id': str(chunk.id),
                    'document_id': str(chunk.document.id),
                    'document_name': chunk.document.name,
                    'document_type': chunk.document.document_type,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'similarity': similarity,
                })
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _extract_text(self, document: Document) -> str:
        """Extract text from document file."""
        file_path = document.file.path
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                text_parts = []
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text_parts.append(page.extract_text())
                return '\n'.join(text_parts)
            except ImportError:
                raise ImportError("PyPDF2 not installed for PDF support")
        
        elif file_ext in ['.doc', '.docx']:
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                raise ImportError("python-docx not installed for Word support")
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '.\n', '\n\n', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'start': start,
                    'end': end,
                })
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
