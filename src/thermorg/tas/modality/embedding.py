# SPDX-License-Identifier: Apache-2.0

"""Embedding-based modality support for text, video, and audio."""

from typing import Optional, Dict, Any, Union
from abc import ABC
import numpy as np
from numpy.typing import NDArray

from .base import BaseModality, ModalityConfig


class EmbeddingModality(BaseModality):
    """Pre-trained embedding support for video/audio/text.
    
    This modality uses pre-trained encoders to extract embeddings:
    - Text: BERT, CLIP text encoder
    - Video: CLIP video encoder, VideoMAE
    - Audio: Wav2Vec, AudioCLIP
    
    Distance metric: Cosine (default for embeddings)
    
    Attributes:
        encoder: Encoder name ('bert', 'clip', 'wav2vec', etc.)
        encoder_model: Optional loaded encoder model
        device: Device for model inference
    
    Example:
        >>> modality = EmbeddingModality(encoder='bert')
        >>> features = modality.extract_features(text_data)
        >>> dist = modality.compute_distance(features[0], features[1])
    """
    
    # Supported encoders and their embedding dimensions
    ENCODER_DIMS = {
        'bert': 768,
        'clip': 512,
        'clip_text': 512,
        'clip_video': 512,
        'wav2vec': 768,
        'audio': 768,
        'video': 512,
        'sentence_transformer': 384,
    }
    
    def __init__(
        self,
        encoder: str = 'clip',
        config: Optional[ModalityConfig] = None,
        device: Optional[str] = None,
    ):
        """Initialize embedding modality.
        
        Args:
            encoder: Encoder name ('bert', 'clip', 'wav2vec', etc.)
            config: Modality configuration
            device: Device for model inference ('cpu', 'cuda', etc.)
        """
        super().__init__(config)
        self.encoder = encoder.lower()
        self.device = device or 'cpu'
        self.encoder_model: Optional[Any] = None
        
        # Set embedding dimension from known encoders
        if self.encoder in self.ENCODER_DIMS:
            self._embedding_dim = self.ENCODER_DIMS[self.encoder]
        
        # Override distance metric to cosine for embeddings
        self.config.distance_metric = 'cosine'
    
    def extract_features(self, data) -> NDArray[np.floating]:
        """Extract features using pre-trained encoder.
        
        Args:
            data: Raw data (text strings, video paths/arrays, audio arrays)
                  For text: List of strings or single string
                  For video: Path to video file or video array
                  For audio: Audio array or path to audio file
                  
        Returns:
            Feature array of shape (n_samples, embedding_dim)
        """
        if self.encoder in ['bert', 'sentence_transformer']:
            return self._extract_text_features(data)
        elif 'clip' in self.encoder:
            return self._extract_clip_features(data)
        elif 'wav2vec' in self.encoder or self.encoder == 'audio':
            return self._extract_audio_features(data)
        elif self.encoder in ['video', 'videomae']:
            return self._extract_video_features(data)
        else:
            return self._extract_generic_features(data)
    
    def _extract_text_features(self, text_data) -> NDArray[np.floating]:
        """Extract text features using BERT or similar.
        
        Args:
            text_data: Text data (string or list of strings)
            
        Returns:
            Feature array
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required for BERT. "
                "Install with: pip install transformers"
            )
        
        if isinstance(text_data, str):
            text_data = [text_data]
        
        # Load model if not cached
        if self.encoder_model is None:
            model_name = 'bert-base-uncased' if self.encoder == 'bert' else 'all-MiniLM-L6-v2'
            self.encoder_model = AutoModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.encoder_model.eval()
        
        features = []
        with torch.no_grad():
            for text in text_data:
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.encoder_model(**inputs)
                # Use [CLS] token embedding
                feat = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(feat)
        
        result = np.vstack(features)
        self._embedding_dim = result.shape[1]
        return result
    
    def _extract_clip_features(self, data) -> NDArray[np.floating]:
        """Extract CLIP features for text or video.
        
        Args:
            data: Text strings or image/video data
            
        Returns:
            Feature array
        """
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "torch and PIL required for CLIP. "
                "Install with: pip install torch torchvision Pillow"
            )
        
        try:
            import clip
        except ImportError:
            raise ImportError(
                "clip library required. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        
        if self.encoder_model is None:
            self.encoder_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.encoder_model.eval()
        
        # Determine if text or video
        if isinstance(data, (str, list)) and (isinstance(data, str) or all(isinstance(d, str) for d in data)):
            # Text input
            if isinstance(data, str):
                data = [data]
            text_tokens = clip.tokenize(data).to(self.device)
            
            with torch.no_grad():
                features = self.encoder_model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            
            result = features.cpu().numpy()
        else:
            # Image/video input
            raise NotImplementedError(
                "Video/image features not yet implemented. "
                "Use text input for CLIP modality."
            )
        
        self._embedding_dim = result.shape[1]
        return result
    
    def _extract_audio_features(self, audio_data) -> NDArray[np.floating]:
        """Extract audio features using Wav2Vec.
        
        Args:
            audio_data: Audio array or path to audio file
            
        Returns:
            Feature array
        """
        try:
            import torch
            import torchaudio
        except ImportError:
            raise ImportError(
                "torchaudio required for Wav2Vec. "
                "Install with: pip install torchaudio"
            )
        
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
        except ImportError:
            raise ImportError(
                "transformers required for Wav2Vec. "
                "Install with: pip install transformers"
            )
        
        if self.encoder_model is None:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.encoder_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        
        self.encoder_model.eval()
        
        # Handle audio input
        if isinstance(audio_data, str):
            waveform, sample_rate = torchaudio.load(audio_data)
        else:
            waveform = torch.tensor(audio_data) if not isinstance(audio_data, torch.Tensor) else audio_data
            sample_rate = 16000  # Default for Wav2Vec
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Process
        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder_model(**inputs)
            # Use mean pooling
            feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        result = feat
        self._embedding_dim = result.shape[1]
        return result
    
    def _extract_video_features(self, video_data) -> NDArray[np.floating]:
        """Extract video features using VideoMAE or similar.
        
        Args:
            video_data: Video file path or video array
            
        Returns:
            Feature array
        """
        # Placeholder for video feature extraction
        raise NotImplementedError(
            "Video feature extraction not yet implemented. "
            "Use CLIP video encoder or VideoMAE."
        )
    
    def _extract_generic_features(self, data) -> NDArray[np.floating]:
        """Generic feature extraction fallback.
        
        Args:
            data: Data array
            
        Returns:
            Feature array (identity if already numeric)
        """
        X = np.asarray(data, dtype=np.float64)
        X = self._ensure_2d(X)
        
        if self.config.normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            X = X / norms
        
        self._embedding_dim = X.shape[1]
        return X
    
    def compute_distance(self, x1: NDArray, x2: NDArray) -> float:
        """Compute cosine distance between two embedding samples.
        
        Args:
            x1: First sample embedding
            x2: Second sample embedding
            
        Returns:
            Cosine distance (1 - cosine_similarity)
        """
        # Flatten if needed
        x1 = x1.flatten()
        x2 = x2.flatten()
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Max distance for zero vectors
        
        cos_sim = np.dot(x1, x2) / (norm1 * norm2)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        return float(1.0 - cos_sim)
    
    def compute_similarity(self, x1: NDArray, x2: NDArray) -> float:
        """Compute cosine similarity between two embedding samples.
        
        Args:
            x1: First sample embedding
            x2: Second sample embedding
            
        Returns:
            Cosine similarity
        """
        return 1.0 - self.compute_distance(x1, x2)
    
    def load_encoder(self, model_name: Optional[str] = None):
        """Pre-load encoder model for faster inference.
        
        Args:
            model_name: Optional custom model name
        """
        if self.encoder == 'bert':
            self._load_bert(model_name)
        elif 'clip' in self.encoder:
            self._load_clip(model_name)
        elif 'wav2vec' in self.encoder:
            self._load_wav2vec(model_name)
    
    def _load_bert(self, model_name: Optional[str] = None):
        """Load BERT model."""
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        model_name = model_name or 'bert-base-uncased'
        self.encoder_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _load_clip(self, model_name: Optional[str] = None):
        """Load CLIP model."""
        import clip
        import torch
        
        model_name = model_name or "ViT-B/32"
        self.encoder_model, self.preprocess = clip.load(model_name, device=self.device)
    
    def _load_wav2vec(self, model_name: Optional[str] = None):
        """Load Wav2Vec model."""
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        
        model_name = model_name or "facebook/wav2vec2-base"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.encoder_model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
    
    def unload(self):
        """Unload encoder model to free memory."""
        self.encoder_model = None
        if hasattr(self, 'tokenizer'):
            self.tokenizer = None
        if hasattr(self, 'preprocess'):
            self.preprocess = None
        if hasattr(self, 'processor'):
            self.processor = None
