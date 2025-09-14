# MILK10k Medical Image Segmentation and Classification Pipeline
# Test version with 20 image limit for validation

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
import nibabel as nib
import SimpleITK as sitk
from collections import Counter, defaultdict
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from skimage import filters, morphology, measure
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# Set up Python path for ConceptModel imports
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')

# Import local ConceptCLIP modules directly
from ConceptModel.modeling_conceptclip import ConceptCLIP
from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor

# ==================== CONFIGURATION ====================

# Your dataset paths (Narval specific)
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_GroundTruth.csv"
OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/outputs"

# Local model paths
SAM2_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2"
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"

# ==================== GPU DETECTION AND SETUP ====================

def setup_gpu_environment():
    """Setup GPU environment with proper error handling"""
    print("=" * 50)
    print("GPU ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB)")
        
        # Set default device
        device = f"cuda:{torch.cuda.current_device()}"
        print(f"Using device: {device}")
        
        # Test GPU allocation
        try:
            test_tensor = torch.randn(10, 10).to(device)
            print("✅ GPU allocation test successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU allocation test failed: {e}")
            print("Falling back to CPU")
            device = "cpu"
    else:
        print("⚠️ CUDA not available. Using CPU.")
        device = "cpu"
        
        # Check Slurm GPU allocation
        slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE', 'Not set')
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        print(f"SLURM_GPUS_ON_NODE: {slurm_gpus}")
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    print("=" * 50)
    return device

# ==================== CACHE AND OFFLINE SETUP ====================

def setup_offline_environment(cache_path: str):
    """Setup offline environment for Hugging Face models"""
    print("=" * 50)
    print("OFFLINE ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Set environment variables for offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1" 
    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    print(f"✅ Offline mode enabled")
    print(f"✅ Cache directory set to: {cache_path}")
    
    # Verify cache directory exists
    cache_path_obj = Path(cache_path)
    if cache_path_obj.exists():
        print(f"✅ Cache directory exists")
        cached_models = list(cache_path_obj.glob("models--*"))
        print(f"✅ Found {len(cached_models)} cached models:")
        for model in cached_models:
            print(f"   - {model.name}")
    else:
        print(f"❌ Cache directory does not exist: {cache_path}")
        
    print("=" * 50)

# ==================== SAM2 MODEL LOADING ====================

def load_local_sam2_model(model_path: str, device: str):
    """Load local SAM2 model"""
    try:
        print(f"Loading SAM2 from local path: {model_path}")
        
        # Import SAM2 components
        sys.path.insert(0, model_path)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Load model configuration
        model_cfg = "sam2_hiera_l.yaml"
        sam2_checkpoint = os.path.join(model_path, "checkpoints", "sam2_hiera_large.pt")
        
        # Build model
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        print(f"SAM2 loaded successfully on {device}")
        return predictor
        
    except Exception as e:
        print(f"Error loading local SAM2: {e}")
        print("Creating dummy SAM2 predictor for testing...")
        return create_dummy_sam_predictor()

def create_dummy_sam_predictor():
    """Create a dummy SAM predictor for testing when SAM2 is not available"""
    class DummySAMPredictor:
        def set_image(self, image):
            self.image = image
            
        def predict(self, point_coords, point_labels, box=None, multimask_output=True):
            h, w = self.image.shape[:2]
            # Create a simple circular mask for testing
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4
            
            Y, X = np.ogrid[:h, :w]
            mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
            
            masks = np.array([mask.astype(np.uint8)])
            scores = np.array([0.8])
            logits = None
            
            return masks, scores, logits
    
    return DummySAMPredictor()

# ==================== LOCAL MODEL LOADING ====================

def load_local_conceptclip_models(model_path: str, cache_path: str, device: str):
    """Load local ConceptCLIP models with offline support"""
    try:
        # Setup offline environment first
        setup_offline_environment(cache_path)
        
        print(f"Loading ConceptCLIP from local path: {model_path}")
        print(f"Using cache directory: {cache_path}")
        
        # Load model with local_files_only to ensure offline mode
        model = ConceptCLIP.from_pretrained(
            model_path,
            local_files_only=True,
            cache_dir=cache_path
        )
        
        # Try to load processor from ConceptCLIP
        try:
            processor = ConceptCLIPProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                cache_dir=cache_path
            )
        except Exception as e:
            print(f"Using simple processor due to error: {e}")
            processor = create_simple_processor()
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        print(f"✅ ConceptCLIP loaded successfully on {device}")
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading local ConceptCLIP: {e}")
        print("This might be due to missing dependencies. Trying fallback...")
        return create_dummy_conceptclip_model(device), create_simple_processor()

def create_dummy_conceptclip_model(device: str):
    """Create a dummy ConceptCLIP model for testing"""
    class DummyConceptCLIP:
        def __init__(self, device):
            self.device = device
            
        def to(self, device):
            self.device = device
            return self
            
        def eval(self):
            return self
            
        def __call__(self, **inputs):
            # Return dummy outputs
            batch_size = inputs['pixel_values'].shape[0] if 'pixel_values' in inputs else 1
            text_size = inputs['input_ids'].shape[0] if 'input_ids' in inputs else 10
            
            return {
                'image_features': torch.randn(batch_size, 512).to(self.device),
                'text_features': torch.randn(text_size, 512).to(self.device),
                'logit_scale': torch.tensor(2.6592).to(self.device)
            }
    
    return DummyConceptCLIP(device)

def create_simple_processor():
    """Create a simple processor for ConceptCLIP"""
    class SimpleProcessor:
        def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
            import torch
            from PIL import Image
            import torchvision.transforms as transforms
            
            result = {}
            
            if images is not None:
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                if isinstance(images, Image.Image):
                    images = [images]
                
                processed = torch.stack([transform(img) for img in images])
                result['pixel_values'] = processed
            
            if text is not None:
                # Simple text encoding
                if isinstance(text, str):
                    text = [text]
                
                # Create dummy tokens for now
                max_length = 77
                result['input_ids'] = torch.randint(0, 1000, (len(text), max_length))
                result['attention_mask'] = torch.ones((len(text), max_length))
            
            return result
    
    return SimpleProcessor()

# ==================== MILK10k DOMAIN CONFIGURATION ====================

@dataclass
class MedicalDomain:
    """Configuration for MILK10k medical imaging domain"""
    name: str
    image_extensions: List[str]
    text_prompts: List[str]
    label_mappings: Dict[str, str]
    preprocessing_params: Dict
    segmentation_strategy: str

MILK10K_DOMAIN = MedicalDomain(
    name="milk10k",
    image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.dcm', '.dicom'],
    text_prompts=[
        'a dermatoscopic image showing actinic keratosis',
        'a dermatoscopic image showing basal cell carcinoma',
        'a dermatoscopic image showing benign proliferation',
        'a dermatoscopic image showing benign keratinocytic lesion',
        'a dermatoscopic image showing dermatofibroma',
        'a dermatoscopic image showing inflammatory condition',
        'a dermatoscopic image showing malignant proliferation',
        'a dermatoscopic image showing melanoma',
        'a dermatoscopic image showing melanocytic nevus',
        'a dermatoscopic image showing squamous cell carcinoma',
        'a dermatoscopic image showing vascular lesion'
    ],
    label_mappings={
        'AKIEC': 'actinic keratosis',
        'BCC': 'basal cell carcinoma',
        'BEN_OTH': 'benign proliferation',
        'BKL': 'benign keratinocytic lesion',
        'DF': 'dermatofibroma',
        'INF': 'inflammatory condition',
        'MAL_OTH': 'malignant proliferation',
        'MEL': 'melanoma',
        'NV': 'melanocytic nevus',
        'SCCKA': 'squamous cell carcinoma',
        'VASC': 'vascular lesion'
    },
    preprocessing_params={'normalize': True, 'enhance_contrast': True},
    class_names=[
        'actinic keratosis',
        'basal cell carcinoma',
        'benign proliferation',
        'benign keratinocytic lesion',
        'dermatofibroma',
        'inflammatory condition',
        'malignant proliferation',
        'melanoma',
        'melanocytic nevus',
        'squamous cell carcinoma',
        'vascular lesion'
    ]
)

print(f"✓ MILK10k domain configured with {len(MILK10K_DOMAIN.class_names)} classes")
print("✓ SECTION: Domain configuration completed successfully")
print("-"*60)


# ==================== MAIN PIPELINE CLASS ====================

class MILK10kPipeline:
    """MILK10k segmentation and classification pipeline"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, output_path: str, 
                 sam2_model_path: str = None, conceptclip_model_path: str = None,
                 cache_path: str = None, max_images: int = None):
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.output_path = Path(output_path)
        self.sam2_model_path = sam2_model_path or SAM2_MODEL_PATH
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.cache_path = cache_path or HUGGINGFACE_CACHE_PATH
        self.domain = MILK10K_DOMAIN
        self.max_images = max_images  # Store max_images limit
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "segmented").mkdir(exist_ok=True)
        (self.output_path / "segmented_for_conceptclip").mkdir(exist_ok=True)
        (self.output_path / "classifications").mkdir(exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        
        # Initialize device with proper setup
        self.device = setup_gpu_environment()
        print(f"Initializing MILK10k pipeline on {self.device}")
        
        # Load models
        self._load_models()
        
        # Load ground truth
        self._load_ground_truth()
        
    def _load_models(self):
        """Load local SAM2 and ConceptCLIP models with device handling"""
        
        # Load local SAM2 with device parameter
        self.sam_predictor = load_local_sam2_model(self.sam2_model_path, self.device)
        
        # Load local ConceptCLIP with cache support
        self.conceptclip_model, self.conceptclip_processor = load_local_conceptclip_models(
            self.conceptclip_model_path, self.cache_path, self.device
        )
        
    def _load_ground_truth(self):
        """Load ground truth annotations"""
        if os.path.exists(self.groundtruth_path):
            self.ground_truth = pd.read_csv(self.groundtruth_path)
            print(f"Loaded ground truth: {len(self.ground_truth)} samples")
            print(f"Ground truth columns: {list(self.ground_truth.columns)}")
        else:
            print(f"Ground truth file not found: {self.groundtruth_path}")
            self.ground_truth = None
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess images for MILK10k dataset"""
        try:
            image_path = Path(image_path)
            ext = image_path.suffix.lower()
            
            if ext in ['.dcm', '.dicom']:
                return self._load_dicom(image_path)
            elif ext in ['.nii', '.nii.gz']:
                return self._load_nifti(image_path)
            else:
                return self._load_standard_image(image_path)
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _load_dicom(self, image_path: Path) -> np.ndarray:
        """Load DICOM images"""
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array.astype(np.float32)
        
        # Normalize
        image = self._normalize_image(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def _load_nifti(self, image_path: Path) -> np.ndarray:
        """Load NIfTI images"""
        nii_img = nib.load(image_path)
        image = nii_img.get_fdata()
        
        # Take middle slice for 3D volumes
        if len(image.shape) == 3:
            mid_slice = image.shape[2] // 2
            image = image[:, :, mid_slice]
        
        # Normalize
        image = self._normalize_image(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def _load_standard_image(self, image_path: Path) -> np.ndarray:
        """Load standard image formats"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast
        if self.domain.preprocessing_params.get('enhance_contrast', False):
            image = self._enhance_contrast(image)
        
        return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
        return (image * 255).astype(np.uint8)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Segment image using SAM2 with device-aware processing"""
        h, w = image.shape[:2]
        
        # Adaptive segmentation strategy
        center_points = self._find_roi_points(image)
        point_labels = np.ones(len(center_points))
        
        # SAM2 segmentation with proper error handling
        try:
            with torch.inference_mode():
                # Device-aware processing
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    # GPU processing with mixed precision
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        self.sam_predictor.set_image(image)
                        masks, scores, _ = self.sam_predictor.predict(
                            point_coords=center_points,
                            point_labels=point_labels,
                            box=None,
                            multimask_output=True
                        )
                else:
                    # CPU processing
                    self.sam_predictor.set_image(image)
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=center_points,
                        point_labels=point_labels,
                        box=None,
                        multimask_output=True
                    )
            
            # Select best mask
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(np.uint8)
            confidence = float(scores[best_idx])
            
            return mask, confidence
            
        except Exception as e:
            print(f"SAM2 segmentation error: {e}")
            # Return empty mask as fallback
            return np.zeros((h, w), dtype=np.uint8), 0.0
    
    def _find_roi_points(self, image: np.ndarray) -> np.ndarray:
        """Find regions of interest for adaptive segmentation"""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Find high-contrast regions
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        
        # Find connected components
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        # Select largest regions as ROI points
        roi_points = []
        for region in sorted(regions, key=lambda x: x.area, reverse=True)[:3]:
            y, x = region.centroid
            roi_points.append([int(x), int(y)])
        
        if not roi_points:
            # Fallback to center
            h, w = image.shape[:2]
            roi_points = [[w // 2, h // 2]]
        
        return np.array(roi_points)
    
    def create_medical_segmentation_outputs(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create medical-standard segmentation visualizations
    Focuses on clinically relevant presentations
    """
    outputs = {}
    h, w = image.shape[:2]
    
    # 1. CLINICAL STANDARD: Semi-transparent colored overlay
    # This is the most common presentation in medical imaging
    overlay_color = (255, 100, 100)  # Soft red for pathology
    alpha = 0.4  # Semi-transparent
    
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = overlay_color
    
    clinical_overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    outputs['clinical_overlay'] = clinical_overlay
    
    # 2. CONTOUR HIGHLIGHTING: Professional medical standard
    contour_image = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use medical-standard colors
    cv2.drawContours(contour_image, contours, -1, (0, 255, 255), 2)  # Cyan outline
    outputs['medical_contour'] = contour_image
    
    # 3. MULTI-COLOR OVERLAY: Different colors for different confidence levels
    if hasattr(self, 'segmentation_confidence'):
        conf = getattr(self, 'segmentation_confidence', 0.8)
        if conf > 0.8:
            color = (255, 0, 0)  # High confidence: Red
        elif conf > 0.6:
            color = (255, 165, 0)  # Medium confidence: Orange  
        else:
            color = (255, 255, 0)  # Low confidence: Yellow
            
        conf_overlay = image.copy()
        conf_mask = np.zeros_like(image)
        conf_mask[mask == 1] = color
        confidence_viz = cv2.addWeighted(image, 0.7, conf_mask, 0.3, 0)
        outputs['confidence_overlay'] = confidence_viz
    
    # 4. ANNOTATED VERSION: With measurement info
    annotated = clinical_overlay.copy()
    
    # Calculate basic measurements
    area_pixels = np.sum(mask)
    area_percentage = (area_pixels / (h * w)) * 100
    
    # Find centroid for annotation placement
    coords = np.where(mask == 1)
    if len(coords[0]) > 0:
        centroid_y = int(np.mean(coords[0]))
        centroid_x = int(np.mean(coords[1]))
        
        # Add annotation text
        annotation_text = f"Area: {area_percentage:.1f}%"
        cv2.putText(annotated, annotation_text, 
                   (centroid_x - 50, centroid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add arrow pointing to centroid
        cv2.arrowedLine(annotated, 
                       (centroid_x - 80, centroid_y - 30),
                       (centroid_x - 10, centroid_y - 15),
                       (255, 255, 255), 2)
    
    outputs['annotated_clinical'] = annotated
    
    # 5. HEATMAP STYLE: For uncertainty visualization
    # Convert mask to heatmap overlay
    heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heatmap_overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    outputs['heatmap_overlay'] = heatmap_overlay
    
    # 6. SIDE-BY-SIDE COMPARISON: Original vs. Segmented
    # Resize for consistent display
    comparison = np.hstack([
        cv2.putText(image.copy(), "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2),
        cv2.putText(clinical_overlay.copy(), "Segmented", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ])
    outputs['clinical_comparison'] = comparison
    
    # 7. CROPPED WITH CONTEXT: Bounding box + padding for context
    coords = np.where(mask == 1)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add substantial padding for medical context
        padding = max(50, min(h, w) // 10)  # Adaptive padding
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        
        # Crop both original and segmented versions
        cropped_original = image[y_min:y_max, x_min:x_max]
        cropped_segmented = clinical_overlay[y_min:y_max, x_min:x_max]
        
        outputs['cropped_context'] = cropped_segmented
        outputs['cropped_original'] = cropped_original
    
    return outputs

def get_priority_outputs_for_conceptclip(self, segmented_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Return the most clinically relevant outputs for ConceptCLIP classification
    """
    priority_outputs = {}
    
    # Priority order for medical classification
    priority_keys = [
        'clinical_overlay',      # Most important: full context with clear segmentation
        'medical_contour',       # Second: boundary emphasis
        'annotated_clinical',    # Third: with measurements
        'cropped_context',       # Fourth: focused region with context
        'confidence_overlay'     # Fifth: confidence-based coloring
    ]
    
    for key in priority_keys:
        if key in segmented_outputs:
            priority_outputs[key] = segmented_outputs[key]
    
    return priority_outputs

def classify_with_medical_priority(self, segmented_outputs: Dict[str, np.ndarray]) -> Dict:
    """
    Enhanced classification focusing on medically relevant visualizations
    """
    # Get priority outputs for classification
    priority_outputs = self.get_priority_outputs_for_conceptclip(segmented_outputs)
    
    results = {}
    
    # Classify each priority output
    for output_type, seg_image in priority_outputs.items():
        if seg_image is not None and seg_image.size > 0:
            seg_pil = Image.fromarray(seg_image.astype(np.uint8))
            
            # Use ConceptCLIP processor
            inputs = self.conceptclip_processor(
                images=seg_pil, 
                text=self.domain.text_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.conceptclip_model(**inputs)
                
                logit_scale = outputs.get("logit_scale", torch.tensor(1.0))
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                
                # Compute similarity scores
                logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)[0]
            
            # Convert to probabilities
            disease_names = [prompt.split(' showing ')[-1] for prompt in self.domain.text_prompts]
            probabilities = {disease_names[i]: float(logits[i]) for i in range(len(disease_names))}
            results[output_type] = probabilities
    
    # Medical-focused ensemble weights
    medical_weights = {
        'clinical_overlay': 0.35,      # Highest weight - standard medical presentation
        'medical_contour': 0.25,       # High weight - clear boundaries
        'annotated_clinical': 0.20,    # Good weight - includes measurements
        'cropped_context': 0.15,       # Moderate weight - focused view
        'confidence_overlay': 0.05     # Low weight - supplementary info
    }
    
    return self._ensemble_predictions(results, medical_weights)
    
    def classify_segmented_image(self, segmented_outputs: Dict[str, np.ndarray]) -> Dict:
        """Classify using local ConceptCLIP on segmented outputs"""
        try:
            results = {}
            
            # Classify each segmented output type
            for output_type, seg_image in segmented_outputs.items():
                if seg_image is not None and seg_image.size > 0:
                    seg_pil = Image.fromarray(seg_image.astype(np.uint8))
                    
                    # Use ConceptCLIP processor
                    inputs = self.conceptclip_processor(
                        images=seg_pil, 
                        text=self.domain.text_prompts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True
                    )
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.conceptclip_model(**inputs)
                        
                        # Extract logits using ConceptCLIP output structure
                        logit_scale = outputs.get("logit_scale", torch.tensor(1.0))
                        image_features = outputs["image_features"]
                        text_features = outputs["text_features"]
                        
                        # Compute similarity scores
                        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)[0]
                    
                    # Convert to probabilities
                    disease_names = [prompt.split(' showing ')[-1] for prompt in self.domain.text_prompts]
                    probabilities = {disease_names[i]: float(logits[i]) for i in range(len(disease_names))}
                    results[output_type] = probabilities
            
            # Ensemble results (weighted average)
            weights = {
                'colored_overlay': 0.3,
                'contour': 0.2,
                'cropped': 0.25,
                'masked_only': 0.15,
                'side_by_side': 0.1
            }
            
            final_probs = self._ensemble_predictions(results, weights)
            return final_probs
            
        except Exception as e:
            print(f"Classification error: {e}")
            return {}
    
    def _ensemble_predictions(self, results: Dict, weights: Dict) -> Dict:
        """Ensemble multiple prediction strategies"""
        if not results:
            return {}
        
        # Get all disease names
        disease_names = list(next(iter(results.values())).keys())
        
        # Weighted average
        final_probs = {}
        for disease in disease_names:
            weighted_sum = 0
            total_weight = 0
            
            for output_type, probs in results.items():
                if output_type in weights and disease in probs:
                    weighted_sum += weights[output_type] * probs[disease]
                    total_weight += weights[output_type]
            
            final_probs[disease] = weighted_sum / total_weight if total_weight > 0 else 0
        
        return final_probs
    
    def get_ground_truth_label(self, img_path: Path) -> Optional[str]:
        """Get ground truth label for image"""
        if self.ground_truth is None:
            return None
        
        img_name = img_path.stem
        
        # Try to find matching row in ground truth
        matching_rows = self.ground_truth[
            self.ground_truth.iloc[:, 0].astype(str).str.contains(img_name, na=False)
        ]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            # Look for the label in subsequent columns
            for col in self.ground_truth.columns[1:]:
                if col in self.domain.label_mappings and row[col] == 1:
                    return self.domain.label_mappings[col]
            
            # If no specific column, check if there's a direct label column
            if 'label' in row:
                return str(row['label'])
        
        return None
    
    def process_dataset(self) -> Dict:
        """Process entire MILK10k dataset or limited subset for testing"""
        print("Starting MILK10k dataset processing...")
        
        # Find all images
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(self.dataset_path.rglob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images in dataset")
        
        # Limit images if max_images is set (for testing)
        if self.max_images:
            image_files = image_files[:self.max_images]
            print(f"⚠️ TEST MODE: Processing only {len(image_files)} images")
            print("=" * 50)
        
        results = []
        format_counter = Counter()
        correct_predictions = 0
        total_with_gt = 0
        
        # Update progress bar description for test mode
        desc = f"Processing {'TEST' if self.max_images else 'MILK10k'} images"
        
        for img_path in tqdm(image_files, desc=desc):
            try:
                # Track file formats
                ext = img_path.suffix.lower()
                format_counter[ext] += 1
                
                # Load and preprocess image
                image = self.preprocess_image(img_path)
                if image is None:
                    continue
                
                # Segment image
                mask, seg_confidence = self.segment_image(image)
                
                # Create segmented outputs for ConceptCLIP
                segmented_outputs = self.create_segmented_outputs(image, mask)
                
                # Save segmented outputs for ConceptCLIP input
                img_name = img_path.stem
                conceptclip_dir = self.output_path / "segmented_for_conceptclip" / img_name
                conceptclip_dir.mkdir(exist_ok=True)
                
                for output_type, seg_image in segmented_outputs.items():
                    if seg_image is not None:
                        output_path = conceptclip_dir / f"{output_type}.png"
                        cv2.imwrite(str(output_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
                
                # Classify using ConceptCLIP
                classification_probs = self.classify_segmented_image(segmented_outputs)
                
                # Get ground truth
                ground_truth = self.get_ground_truth_label(img_path)
                
                # Get prediction
                if classification_probs:
                    predicted_disease = max(classification_probs, key=classification_probs.get)
                    prediction_confidence = classification_probs[predicted_disease]
                else:
                    predicted_disease = "unknown"
                    prediction_confidence = 0.0
                
                # Check accuracy if ground truth available
                if ground_truth:
                    total_with_gt += 1
                    if ground_truth == predicted_disease:
                        correct_predictions += 1
                
                # Save results
                result = {
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'predicted_disease': predicted_disease,
                    'prediction_confidence': prediction_confidence,
                    'segmentation_confidence': seg_confidence,
                    'ground_truth': ground_truth,
                    'correct': ground_truth == predicted_disease if ground_truth else None,
                    'segmented_outputs_dir': str(conceptclip_dir),
                    'classification_probabilities': classification_probs,
                    'device_used': self.device,
                    'cache_used': self.cache_path,
                    'test_mode': self.max_images is not None
                }
                
                results.append(result)
                
                # Progress indicator
                status = "✓" if result['correct'] else ("✗" if ground_truth else "-")
                print(f"{status} {img_name}: {predicted_disease} ({prediction_confidence:.2%}) [Device: {self.device}]")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate accuracy
        accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
        
        # Generate report
        report = self._generate_comprehensive_report(results, format_counter, accuracy, total_with_gt)
        
        # Add test mode info to report
        if self.max_images:
            report['test_mode'] = {
                'enabled': True,
                'images_limit': self.max_images,
                'images_processed': len(results)
            }
        
        # Save results
        self._save_results(results, report)
        
        return report
    
    def _generate_comprehensive_report(self, results: List[Dict], format_counter: Counter, 
                                     accuracy: float, total_with_gt: int) -> Dict:
        """Generate comprehensive processing report"""
        
        # Basic statistics
        total_processed = len(results)
        successful_segmentations = sum(1 for r in results if r['segmentation_confidence'] > 0.5)
        successful_classifications = sum(1 for r in results if r['prediction_confidence'] > 0.1)
        
        # Prediction distribution
        predictions = [r['predicted_disease'] for r in results]
        prediction_counts = Counter(predictions)
        
        # Confidence statistics
        seg_confidences = [r['segmentation_confidence'] for r in results]
        pred_confidences = [r['prediction_confidence'] for r in results]
        
        # Device statistics
        device_used = results[0]['device_used'] if results else "unknown"
        cache_used = results[0]['cache_used'] if results else "unknown"
        
        report = {
            'system_info': {
                'device_used': device_used,
                'cache_directory': cache_used,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'offline_mode': os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            },
            'dataset_info': {
                'total_images_found': total_processed,
                'file_formats': dict(format_counter),
                'total_with_ground_truth': total_with_gt
            },
            'processing_stats': {
                'successful_segmentations': successful_segmentations,
                'successful_classifications': successful_classifications,
                'segmentation_success_rate': successful_segmentations / total_processed if total_processed > 0 else 0,
                'classification_success_rate': successful_classifications / total_processed if total_processed > 0 else 0
            },
            'accuracy_metrics': {
                'overall_accuracy': accuracy,
                'correct_predictions': sum(1 for r in results if r['correct']),
                'total_evaluated': total_with_gt
            },
            'predictions': {
                'distribution': dict(prediction_counts),
                'most_common': prediction_counts.most_common(5)
            },
            'confidence_stats': {
                'segmentation': {
                    'mean': np.mean(seg_confidences) if seg_confidences else 0,
                    'std': np.std(seg_confidences) if seg_confidences else 0,
                    'min': np.min(seg_confidences) if seg_confidences else 0,
                    'max': np.max(seg_confidences) if seg_confidences else 0
                },
                'classification': {
                    'mean': np.mean(pred_confidences) if pred_confidences else 0,
                    'std': np.std(pred_confidences) if pred_confidences else 0,
                    'min': np.min(pred_confidences) if pred_confidences else 0,
                    'max': np.max(pred_confidences) if pred_confidences else 0
                }
            }
        }
        
        return report
    
    def _save_results(self, results: List[Dict], report: Dict):
        """Save results and report"""
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = self.output_path / "reports" / "detailed_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save report
        report_path = self.output_path / "reports" / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary visualization
        self._create_summary_plots(results_df, report)
        
        print(f"\nResults saved to: {self.output_path}")
        print(f"Segmented outputs for ConceptCLIP: {self.output_path / 'segmented_for_conceptclip'}")
        print(f"Detailed results: {results_path}")
        print(f"Processing report: {report_path}")
    
    def _create_summary_plots(self, results_df: pd.DataFrame, report: Dict):
        """Create summary visualization plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction distribution
        pred_counts = report['predictions']['distribution']
        axes[0,0].bar(pred_counts.keys(), pred_counts.values())
        axes[0,0].set_title('Disease Prediction Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Confidence distributions
        axes[0,1].hist([results_df['segmentation_confidence'], results_df['prediction_confidence']], 
                      bins=30, alpha=0.7, label=['Segmentation', 'Classification'])
        axes[0,1].set_title('Confidence Distributions')
        axes[0,1].legend()
        
        # 3. Accuracy by confidence level
        if 'ground_truth' in results_df.columns:
            results_with_gt = results_df.dropna(subset=['ground_truth'])
            if len(results_with_gt) > 0:
                conf_bins = np.linspace(0, 1, 11)
                accuracies = []
                for i in range(len(conf_bins)-1):
                    mask = ((results_with_gt['prediction_confidence'] >= conf_bins[i]) & 
                           (results_with_gt['prediction_confidence'] < conf_bins[i+1]))
                    if mask.sum() > 0:
                        acc = results_with_gt[mask]['correct'].mean()
                        accuracies.append(acc)
                    else:
                        accuracies.append(0)
                
                axes[1,0].plot(conf_bins[:-1], accuracies, marker='o')
                axes[1,0].set_title('Accuracy vs Prediction Confidence')
                axes[1,0].set_xlabel('Prediction Confidence')
                axes[1,0].set_ylabel('Accuracy')
        
        # 4. Processing success rates
        success_data = [
            report['processing_stats']['segmentation_success_rate'],
            report['processing_stats']['classification_success_rate'],
            report['accuracy_metrics']['overall_accuracy']
        ]
        axes[1,1].bar(['Segmentation', 'Classification', 'Overall Accuracy'], success_data)
        axes[1,1].set_title('Processing Success Rates')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plot_path = self.output_path / "visualizations" / "summary_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plots saved to: {plot_path}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function with argument parsing"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MILK10k Medical Image Processing Pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode (20 images only)')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process')
    args = parser.parse_args()
    
    # Determine max_images
    max_images = None
    if args.test:
        max_images = 20
    elif args.max_images:
        max_images = args.max_images
    
    print("="*60)
    print("MILK10K MEDICAL IMAGE PROCESSING PIPELINE")
    print("Updated with Local Cache Support")
    if max_images:
        print(f"🔬 TEST MODE: Processing {max_images} images only")
    print("="*60)
    
    # Initialize pipeline with local models and cache
    pipeline = MILK10kPipeline(
        dataset_path=DATASET_PATH,
        groundtruth_path=GROUNDTRUTH_PATH,
        output_path=OUTPUT_PATH,
        sam2_model_path=SAM2_MODEL_PATH,
        conceptclip_model_path=CONCEPTCLIP_MODEL_PATH,
        cache_path=HUGGINGFACE_CACHE_PATH,
        max_images=max_images
    )
    
    # Process dataset
    report = pipeline.process_dataset()
    
    # Print summary
    print("\n" + "="*50)
    if max_images:
        print(f"MILK10K TEST RUN COMPLETE ({max_images} images)")
    else:
        print("MILK10K PROCESSING COMPLETE")
    print("="*50)
    print(f"Device used: {report['system_info']['device_used']}")
    print(f"Cache directory: {report['system_info']['cache_directory']}")
    print(f"Offline mode: {report['system_info']['offline_mode']}")
    print(f"Total images processed: {report['dataset_info']['total_images_found']}")
    print(f"Successful segmentations: {report['processing_stats']['successful_segmentations']}")
    print(f"Successful classifications: {report['processing_stats']['successful_classifications']}")
    
    if report['accuracy_metrics']['total_evaluated'] > 0:
        print(f"Overall accuracy: {report['accuracy_metrics']['overall_accuracy']:.2%}")
    
    if 'test_mode' in report:
        print(f"\n⚠️ TEST MODE RESULTS - Limited to {report['test_mode']['images_limit']} images")
        print("Run without --test flag to process entire dataset")
    
    print(f"\nSegmented outputs for ConceptCLIP saved to:")
    print(f"{OUTPUT_PATH}/segmented_for_conceptclip/")
    print("\nEach image has multiple segmented versions:")
    print("- colored_overlay.png (main input for ConceptCLIP)")
    print("- contour.png (boundary highlighted)")
    print("- cropped.png (region of interest)")
    print("- masked_only.png (segmented region only)")
    print("- side_by_side.png (comparison view)")

if __name__ == "__main__":
    main()
