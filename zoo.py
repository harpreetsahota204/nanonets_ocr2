import logging
import sys
import io
import warnings
from contextlib import contextmanager
from typing import Union

from PIL import Image
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem
from fiftyone.utils.torch import GetItem

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

@contextmanager
def suppress_output():
    """Suppress stdout, stderr, warnings, and transformers logging."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Suppress transformers logging
    transformers_logger = logging.getLogger("transformers")
    old_transformers_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        transformers_logger.setLevel(old_transformers_level)



def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NanoNetsOCRGetItem(GetItem):
    """GetItem transform for batching NanoNets-OCR inference."""
    
    @property
    def required_keys(self):
        """Fields required from each sample."""
        return ["filepath"]
    
    def __call__(self, sample_dict):
        """Load and return PIL Image from filepath.
        
        Args:
            sample_dict: Dictionary containing sample fields
            
        Returns:
            PIL.Image: Loaded image in RGB format
        """
        filepath = sample_dict["filepath"]
        image = Image.open(filepath).convert("RGB")
        return image


class NanoNetsOCR(Model, SupportsGetItem):
    """FiftyOne model for NanoNets-OCR vision-language tasks with batching support.
    
    Simple OCR model that extracts text from documents using vision-language processing.
    Supports efficient batch processing for faster inference on large datasets.
    
    Automatically selects optimal dtype based on hardware:
    - bfloat16 for CUDA devices with compute capability 8.0+ (Ampere and newer)
    - float16 for older CUDA devices
    - float32 for CPU/MPS devices
    
    Args:
        model_path: HuggingFace model ID or local path (default: "nanonets/Nanonets-OCR2-3B")
        custom_prompt: Custom prompt for OCR task (optional)
        max_new_tokens: Maximum tokens to generate (default: 15000)
        batch_size: Default batch size for inference (default: 4)
        torch_dtype: Override automatic dtype selection
    """
    
    def __init__(
        self,
        model_path: str = "nanonets/Nanonets-OCR2-3B",
        custom_prompt: str = None,
        max_new_tokens: int = 15000,
        batch_size: int = 4,
        torch_dtype: torch.dtype = None,
        **kwargs
    ):
        SupportsGetItem.__init__(self) 
        self.model_path = model_path
        self._custom_prompt = custom_prompt
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._preprocess = False  # Preprocessing happens in GetItem
        
        # Device setup
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Dtype selection
        if torch_dtype is not None:
            self.dtype = torch_dtype
        elif self.device == "cuda":
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            logger.info(f"Using {self.dtype} dtype (compute capability {capability[0]}.{capability[1]})")
        else:
            self.dtype = torch.float32
            logger.info(f"Using float32 dtype for {self.device}")
        
        # Load model, tokenizer, and processor
        logger.info(f"Loading NanoNets-OCR from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        self.model = self.model.eval()
        
        logger.info("NanoNets-OCR model loaded successfully")
    
    @property
    def media_type(self):
        """The media type processed by this model."""
        return "image"
    
    @property
    def supports_batching(self):
        """Whether this model supports batch processing."""
        return True
    
    @property
    def preprocess(self):
        """Whether preprocessing should be applied."""
        return self._preprocess
    
    @preprocess.setter
    def preprocess(self, value):
        """Set preprocessing flag."""
        self._preprocess = value
    
    @property
    def has_collate_fn(self):
        """Whether this model provides a custom collate function."""
        return False  # Use default collation
    
    @property
    def collate_fn(self):
        """Custom collate function for the DataLoader."""
        return None  # Not used
    
    @property
    def ragged_batches(self):
        """Whether this model supports batches with varying sizes."""
        return True  # PIL Images can have different dimensions
    
    def get_item(self):
        """Return the GetItem transform for batching support."""
        return NanoNetsOCRGetItem()
    
    def build_get_item(self, field_mapping=None):
        """Build the GetItem transform for batching.
        
        Args:
            field_mapping: Optional field mapping configuration
            
        Returns:
            NanoNetsOCRGetItem: GetItem transform instance
        """
        return NanoNetsOCRGetItem(field_mapping=field_mapping)
    
    def predict_all(self, images, preprocess=None):
        """Batch prediction for multiple images.
        
        Args:
            images: List of PIL Images to process
            preprocess: Whether to preprocess (convert numpy to PIL)
            
        Returns:
            List[str]: List of extracted text from documents
        """
        # Use instance preprocess flag if not specified
        if preprocess is None:
            preprocess = self._preprocess
        
        # Preprocess if needed (convert numpy to PIL)
        if preprocess:
            pil_images = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                pil_images.append(img)
            images = pil_images
        
        # Use custom prompt if provided, otherwise use default
        prompt = self._custom_prompt if self._custom_prompt else DEFAULT_PROMPT
        
        # Prepare batch of messages (one per image)
        all_messages = []
        for image in images:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},  # PIL Image directly
                    {"type": "text", "text": prompt},
                ]},
            ]
            all_messages.append(messages)
        
        # Run batch inference with suppressed output
        with suppress_output():
            # Apply chat template to all messages
            texts = [
                self.processor.apply_chat_template(
                    msg, 
                    tokenize=False, 
                    add_generation_prompt=True
                ) 
                for msg in all_messages
            ]
            
            # Process batch of inputs
            inputs = self.processor(
                text=texts,        # List of texts
                images=images,     # List of PIL Images
                padding=True,      # Key for batching!
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Batch generation
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                do_sample=False
            )
            
            # Decode only the generated tokens (excluding input)
            batch_size = len(images)
            generated_ids = [
                output_ids[i][len(inputs.input_ids[i]):] 
                for i in range(batch_size)
            ]
            
            # Batch decode to text
            results = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
        
        return results
    
    def _predict(self, image: Image.Image, sample) -> str:
        """Process image through NanoNets-OCR.
        
        Args:
            image: PIL Image to process
            sample: FiftyOne sample (has filepath attribute)
        
        Returns:
            str: Extracted text from the document
        """
        # Use custom prompt if provided, otherwise use default
        prompt = self._custom_prompt if self._custom_prompt else DEFAULT_PROMPT
        
        # Get the image path from the sample
        image_path = sample.filepath if sample else "temp_image.jpg"
        
        # Prepare messages in the chat format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        # Run inference with suppressed output
        with suppress_output():
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate output
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                do_sample=False
            )
            
            # Decode only the generated tokens (excluding input)
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            # Decode to text
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            result = output_text[0]
        
        return result
    
    def predict(self, image, sample=None):
        """Process an image with NanoNets-OCR.
        
        Args:
            image: PIL Image or numpy array to process
            sample: FiftyOne sample containing the image filepath
        
        Returns:
            str: Extracted text from the document
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)
