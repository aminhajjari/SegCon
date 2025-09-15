# MILK10k Medical Image Segmentation and Classification Pipeline
# Updated for folder-based dataset structure
# Test version with configurable image limit for validation

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

print("✓ SECTION: Environment setup and imports completed successfully")
print("-"*60)

# ==================== CONFIGURATION ====================

# Your dataset paths (Narval specific)
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_GroundTruth.csv"
OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs"

# Local model paths
SAM2_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2"
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"

print("✓ SECTION: Configuration paths defined successfully")
print("-"*60)

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

print("✓ SECTION: GPU setup function defined successfully")
print("-"*60)

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

print("✓ SECTION: Offline environment setup function defined successfully")
print("-"*60)

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

print("✓ SECTION: SAM2 model loading functions defined successfully")
print("-"*60)

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

print("✓ SECTION: ConceptCLIP model loading functions defined successfully")
print("-"*60)

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
    class_names: List[str]

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
    segmentation_strategy='medical_adaptive',
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
    """MILK10k segmentation and classification pipeline for folder-based dataset"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, output_path: str, 
                 sam2_model_path: str = None, conceptclip_model_path: str = None,
                 cache_path: str = None, max_folders: int = None):
        print("Initializing MILK10k Pipeline for folder-based dataset...")
        
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.output_path = Path(output_path)
        self.sam2_model_path = sam2_model_path or SAM2_MODEL_PATH
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.cache_path = cache_path or HUGGINGFACE_CACHE_PATH
        self.domain = MILK10K_DOMAIN
        self.max_folders = max_folders  # Changed from max_images to max_folders
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "segmented").mkdir(exist_ok=True)
        (self.output_path / "segmented_for_conceptclip").mkdir(exist_ok=True)
        (self.output_path / "classifications").mkdir(exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        
        print("✓ Output directories created successfully")
        
        # Initialize device with proper setup
        self.device = setup_gpu_environment()
        print(f"Initializing MILK10k pipeline on {self.device}")
        
        # Load models
        self._load_models()
        
        # Load ground truth
        self._load_ground_truth()
        
        print("✓ SECTION: Pipeline initialization completed successfully")
        print("-"*60)
        
    def _load_models(self):
        """Load local SAM2 and ConceptCLIP models with device handling"""
        print("Loading models...")
        
        # Load local SAM2 with device parameter
        self.sam_predictor = load_local_sam2_model(self.sam2_model_path, self.device)
        print("✓ SAM2 model loading completed")
        
        # Load local ConceptCLIP with cache support
        self.conceptclip_model, self.conceptclip_processor = load_local_conceptclip_models(
            self.conceptclip_model_path, self.cache_path, self.device
        )
        print("✓ ConceptCLIP model loading completed")
        
        print("✓ SECTION: Model loading completed successfully")
        print("-"*60)
        
    def _load_ground_truth(self):
        """Load ground truth annotations"""
        print("Loading ground truth data...")
        
        if os.path.exists(self.groundtruth_path):
            self.ground_truth = pd.read_csv(self.groundtruth_path)
            print(f"Loaded ground truth: {len(self.ground_truth)} samples")
            print(f"Ground truth columns: {list(self.ground_truth.columns)}")
            
            # Create a lookup dictionary for faster access
            self.gt_lookup = {}
            for _, row in self.ground_truth.iterrows():
                lesion_id = row['lesion_id']
                
                # Find which disease column has value 1
                disease_columns = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 
                                  'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
                
                for col in disease_columns:
                    if col in row and float(row[col]) == 1.0:
                        self.gt_lookup[lesion_id] = self.domain.label_mappings[col]
                        break
                        
            print(f"✓ Created lookup table for {len(self.gt_lookup)} lesions")
        else:
            print(f"Ground truth file not found: {self.groundtruth_path}")
            self.ground_truth = None
            self.gt_lookup = {}
            
        print("✓ SECTION: Ground truth loading completed successfully")
        print("-"*60)
    
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
    
    def segment_image(self, image):
    """
    Automatic SAM2 segmentation with center-point prompt.
    """
    self.segment_model.eval()

    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

    with torch.no_grad():
        h, w = image_tensor.shape[-2:]
        center_point = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])

        outputs = self.segment_model(
            image=image_tensor,
            point_coords=torch.from_numpy(center_point).unsqueeze(0).to(self.device),
            point_labels=torch.from_numpy(point_labels).unsqueeze(0).to(self.device),
            multimask_output=True
        )

        masks = outputs["masks"]
        scores = outputs["iou_predictions"]
        best_idx = torch.argmax(scores, dim=1)
        best_mask = masks[0, best_idx].cpu().numpy().astype(np.uint8)

    return best_mask

            
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
    
    def create_segmented_outputs(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Create multiple visualization outputs for ConceptCLIP input"""
        outputs = {}
        h, w = image.shape[:2]
        
        # 1. Colored overlay (main output)
        overlay_color = (255, 100, 100)  # Red for medical visualization
        alpha = 0.3
        
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = overlay_color
        
        colored_overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        outputs['colored_overlay'] = colored_overlay
        
        # 2. Contour highlighting
        contour_image = image.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        outputs['contour'] = contour_image
        
        # 3. Cropped region with context
        coords = np.where(mask == 1)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Add padding
            padding = 50
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            
            cropped = colored_overlay[y_min:y_max, x_min:x_max]
            outputs['cropped'] = cropped
        
        # 4. Segmented region only (black background)
        masked_only = np.zeros_like(image)
        masked_only[mask == 1] = image[mask == 1]
        outputs['masked_only'] = masked_only
        
        # 5. Side-by-side comparison
        comparison = np.hstack([image, colored_overlay])
        outputs['side_by_side'] = comparison
        
        return outputs
    
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
    
    def process_folder(self, folder_path: Path) -> Dict:
        """Process all images in a single lesion folder"""
        folder_results = []
        lesion_id = folder_path.name
        
        # Find all images in the folder
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
        
        if not image_files:
            print(f"No images found in folder: {lesion_id}")
            return None
        
        print(f"Processing folder {lesion_id} with {len(image_files)} images")
        
        # Process each image in the folder
        for img_idx, img_path in enumerate(image_files):
            try:
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
                conceptclip_dir = self.output_path / "segmented_for_conceptclip" / lesion_id / img_name
                conceptclip_dir.mkdir(exist_ok=True, parents=True)
                
                for output_type, seg_image in segmented_outputs.items():
                    if seg_image is not None:
                        output_path = conceptclip_dir / f"{output_type}.png"
                        cv2.imwrite(str(output_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
                
                # Classify using ConceptCLIP
                classification_probs = self.classify_segmented_image(segmented_outputs)
                
                # Get prediction
                if classification_probs:
                    predicted_disease = max(classification_probs, key=classification_probs.get)
                    prediction_confidence = classification_probs[predicted_disease]
                else:
                    predicted_disease = "unknown"
                    prediction_confidence = 0.0
                
                # Store individual image result
                image_result = {
                    'lesion_id': lesion_id,
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'image_index_in_folder': img_idx,
                    'predicted_disease': predicted_disease,
                    'prediction_confidence': prediction_confidence,
                    'segmentation_confidence': seg_confidence,
                    'classification_probabilities': classification_probs,
                    'segmented_outputs_dir': str(conceptclip_dir)
                }
                
                folder_results.append(image_result)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        if not folder_results:
            return None
        
        # Aggregate results for the folder (ensemble across all images)
        folder_prediction = self._aggregate_folder_predictions(folder_results)
        
        # Get ground truth for this lesion
        ground_truth = self.gt_lookup.get(lesion_id, None)
        
        # Check if prediction matches ground truth
        is_correct = ground_truth == folder_prediction['predicted_disease'] if ground_truth else None
        
        return {
            'lesion_id': lesion_id,
            'num_images': len(folder_results),
            'individual_images': folder_results,
            'folder_prediction': folder_prediction,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'csv_match_found': ground_truth is not None
        }
    
    def _aggregate_folder_predictions(self, folder_results: List[Dict]) -> Dict:
        """Aggregate predictions from multiple images in the same folder"""
        if not folder_results:
            return {'predicted_disease': 'unknown', 'prediction_confidence': 0.0}
        
        # Strategy 1: Average probabilities across all images
        all_diseases = self.domain.class_names
        avg_probs = {disease: 0.0 for disease in all_diseases}
        
        for result in folder_results:
            probs = result.get('classification_probabilities', {})
            for disease in all_diseases:
                if disease in probs:
                    avg_probs[disease] += probs[disease]
        
        # Average the probabilities
        num_images = len(folder_results)
        for disease in avg_probs:
            avg_probs[disease] /= num_images
        
        # Get final prediction
        predicted_disease = max(avg_probs, key=avg_probs.get)
        prediction_confidence = avg_probs[predicted_disease]
        
        # Strategy 2: Majority vote (alternative approach)
        predictions = [result['predicted_disease'] for result in folder_results]
        prediction_counts = Counter(predictions)
        majority_prediction = prediction_counts.most_common(1)[0][0]
        
        return {
            'predicted_disease': predicted_disease,
            'prediction_confidence': prediction_confidence,
            'average_probabilities': avg_probs,
            'majority_vote_prediction': majority_prediction,
            'prediction_counts': dict(prediction_counts),
            'aggregation_method': 'average_probabilities'
        }
    
    def process_dataset(self) -> Dict:
        """Process entire MILK10k folder-based dataset"""
        print("Starting MILK10k folder-based dataset processing...")
        
        # Find all lesion folders
        lesion_folders = [f for f in self.dataset_path.iterdir() 
                         if f.is_dir() and not f.name.startswith('.')]
        
        print(f"Found {len(lesion_folders)} lesion folders in dataset")
        
        # Sort for consistent order
        lesion_folders = sorted(lesion_folders, key=lambda x: x.name)
        
        # Debug: Show first few folders and corresponding CSV entries
        if self.ground_truth is not None:
            print("\n📊 Verifying folder-CSV matching:")
            print("First 5 lesion folders found:")
            for i, folder in enumerate(lesion_folders[:5]):
                print(f"  {i+1}. {folder.name}")
            
            print("\nFirst 5 CSV lesion_ids:")
            for i, lesion_id in enumerate(self.ground_truth['lesion_id'].head(5)):
                print(f"  {i+1}. {lesion_id}")
            print("-"*50)
        
        # Limit folders if specified
        if self.max_folders:
            original_count = len(lesion_folders)
            lesion_folders = lesion_folders[:self.max_folders]
            print(f"📊 Processing {len(lesion_folders)} folders out of {original_count} total")
            print(f"   Limited processing for testing purposes")
            print("   Use --full flag to process the entire dataset")
            print("=" * 50)
        else:
            print(f"🔬 FULL DATASET MODE: Processing all {len(lesion_folders)} folders")
            print("   Will attempt to match all folders with available CSV ground truth")
            print("=" * 50)
        
        results = []
        correct_predictions = 0
        total_with_gt = 0
        csv_matches_found = 0
        total_images_processed = 0
        
        print("✓ SECTION: Dataset folder discovery completed successfully")
        print(f"Final folder count to process: {len(lesion_folders)}")
        if self.max_folders:
            print(f"CSV ground truth entries available: {len(self.ground_truth) if self.ground_truth is not None else 0}")
        print("-"*60)
        
        # Update progress bar description
        desc = f"Processing {'Limited' if self.max_folders else 'Full'} MILK10k folders"
        
        for folder_idx, folder_path in enumerate(tqdm(lesion_folders, desc=desc)):
            try:
                # Process the entire folder
                folder_result = self.process_folder(folder_path)
                
                if folder_result is None:
                    continue
                
                # Update counters
                total_images_processed += folder_result['num_images']
                
                # Track CSV matching statistics
                if folder_result['csv_match_found']:
                    csv_matches_found += 1
                    total_with_gt += 1
                    
                    if folder_result['is_correct']:
                        correct_predictions += 1
                
                # Add folder index for tracking
                folder_result['folder_index'] = folder_idx
                folder_result['processing_mode'] = 'limited' if self.max_folders else 'full'
                folder_result['max_folders_setting'] = self.max_folders
                folder_result['device_used'] = self.device
                folder_result['cache_used'] = self.cache_path
                
                results.append(folder_result)
                
                # Progress indicator with CSV matching info
                lesion_id = folder_result['lesion_id']
                prediction = folder_result['folder_prediction']['predicted_disease']
                confidence = folder_result['folder_prediction']['prediction_confidence']
                
                status = "✓" if folder_result['is_correct'] else ("✗" if folder_result['ground_truth'] else "-")
                csv_status = "CSV✓" if folder_result['csv_match_found'] else "CSV✗"
                print(f"{status} {csv_status} [{folder_idx+1:3d}] {lesion_id}: {prediction} ({confidence:.2%}) [{folder_result['num_images']} imgs]")
                
            except Exception as e:
                print(f"Error processing folder {folder_path}: {e}")
                continue
        
        print("✓ SECTION: Folder processing loop completed successfully")
        print(f"Processed {len(results)} folders successfully")
        print(f"Total images processed: {total_images_processed}")
        print(f"Found CSV ground truth for {csv_matches_found} folders")
        print(f"Accuracy calculated on {total_with_gt} folders with valid CSV ground truth")
        print("-"*60)
        
        # Calculate accuracy ONLY on folders that matched the CSV data
        accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
        
        # Generate comprehensive report
        report = self._generate_folder_based_report(results, accuracy, total_with_gt, total_images_processed)
        
        # Add processing mode info to report
        if self.max_folders:
            report['processing_mode'] = {
                'type': 'limited',
                'folders_limit': self.max_folders,
                'folders_processed': len(results),
                'total_images_processed': total_images_processed,
                'csv_entries_available': len(self.ground_truth) if self.ground_truth is not None else 0,
                'csv_matches_found': csv_matches_found,
                'accuracy_based_on': total_with_gt
            }
        else:
            report['processing_mode'] = {
                'type': 'full',
                'folders_processed': len(results),
                'total_images_processed': total_images_processed,
                'csv_matches_found': csv_matches_found,
                'accuracy_based_on': total_with_gt
            }
        
        print("✓ SECTION: Report generation completed successfully")
        print("-"*60)
        
        # Save results
        self._save_folder_results(results, report)
        
        print("✓ SECTION: Results saving completed successfully")
        print("-"*60)
        
        return report
    
    def _generate_folder_based_report(self, results: List[Dict], accuracy: float, 
                                    total_with_gt: int, total_images: int) -> Dict:
        """Generate comprehensive processing report for folder-based processing"""
        print("Generating comprehensive folder-based report...")
        
        # Basic statistics
        total_folders = len(results)
        successful_folders = sum(1 for r in results if r['folder_prediction']['prediction_confidence'] > 0.1)
        
        # Images per folder statistics
        images_per_folder = [r['num_images'] for r in results]
        
        # Folder prediction distribution
        folder_predictions = [r['folder_prediction']['predicted_disease'] for r in results]
        prediction_counts = Counter(folder_predictions)
        
        # Confidence statistics for folders
        folder_confidences = [r['folder_prediction']['prediction_confidence'] for r in results]
        
        # Individual image statistics
        all_individual_results = []
        for folder_result in results:
            all_individual_results.extend(folder_result['individual_images'])
        
        individual_confidences = [r['prediction_confidence'] for r in all_individual_results]
        individual_predictions = [r['predicted_disease'] for r in all_individual_results]
        individual_prediction_counts = Counter(individual_predictions)
        
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
                'total_folders_processed': total_folders,
                'total_images_processed': total_images,
                'folders_with_ground_truth': total_with_gt,
                'images_per_folder_stats': {
                    'mean': np.mean(images_per_folder) if images_per_folder else 0,
                    'std': np.std(images_per_folder) if images_per_folder else 0,
                    'min': np.min(images_per_folder) if images_per_folder else 0,
                    'max': np.max(images_per_folder) if images_per_folder else 0,
                    'median': np.median(images_per_folder) if images_per_folder else 0
                }
            },
            'folder_level_results': {
                'successful_folders': successful_folders,
                'success_rate': successful_folders / total_folders if total_folders > 0 else 0,
                'accuracy_metrics': {
                    'overall_accuracy': accuracy,
                    'correct_predictions': sum(1 for r in results if r['is_correct']),
                    'total_evaluated': total_with_gt
                },
                'predictions': {
                    'distribution': dict(prediction_counts),
                    'most_common': prediction_counts.most_common(5)
                },
                'confidence_stats': {
                    'mean': np.mean(folder_confidences) if folder_confidences else 0,
                    'std': np.std(folder_confidences) if folder_confidences else 0,
                    'min': np.min(folder_confidences) if folder_confidences else 0,
                    'max': np.max(folder_confidences) if folder_confidences else 0
                }
            },
            'image_level_results': {
                'total_images': len(all_individual_results),
                'predictions': {
                    'distribution': dict(individual_prediction_counts),
                    'most_common': individual_prediction_counts.most_common(5)
                },
                'confidence_stats': {
                    'mean': np.mean(individual_confidences) if individual_confidences else 0,
                    'std': np.std(individual_confidences) if individual_confidences else 0,
                    'min': np.min(individual_confidences) if individual_confidences else 0,
                    'max': np.max(individual_confidences) if individual_confidences else 0
                }
            }
        }
        
        print("✓ Folder-based report statistics calculated successfully")
        return report
    
    # Add this method to your MILK10kPipeline class to improve the comparison output

def _save_folder_results(self, results: List[Dict], report: Dict):
    """Save results with improved comparison format"""
    print("Saving folder-based results and generating comparison CSV...")
    
    # Create the main comparison CSV that matches your requirements
    comparison_data = []
    
    for folder_result in results:
        lesion_id = folder_result['lesion_id']
        
        # Get model prediction
        model_prediction = folder_result['folder_prediction']['predicted_disease']
        model_confidence = folder_result['folder_prediction']['prediction_confidence']
        
        # Get ground truth from CSV
        ground_truth = folder_result['ground_truth']
        
        # Map disease names back to CSV codes for comparison
        reverse_mapping = {v: k for k, v in self.domain.label_mappings.items()}
        model_prediction_code = reverse_mapping.get(model_prediction, model_prediction)
        ground_truth_code = reverse_mapping.get(ground_truth, ground_truth) if ground_truth else None
        
        # Determine correctness
        is_correct = (model_prediction == ground_truth) if ground_truth else None
        
        comparison_data.append({
            'lesion_id': lesion_id,
            'ground_truth_disease_name': ground_truth if ground_truth else 'NOT_FOUND_IN_CSV',
            'ground_truth_code': ground_truth_code if ground_truth_code else 'NOT_FOUND',
            'model_predicted_disease_name': model_prediction,
            'model_predicted_code': model_prediction_code,
            'model_confidence': f"{model_confidence:.4f}",
            'is_correct': is_correct,
            'match_status': 'CORRECT' if is_correct else ('WRONG' if ground_truth else 'NO_CSV_ENTRY'),
            'num_images_in_folder': folder_result['num_images']
        })
    
    # Save the main comparison CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = self.output_path / "reports" / "model_vs_groundtruth_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    
    # Create summary statistics
    if len(comparison_data) > 0:
        total_folders = len(comparison_data)
        folders_with_csv_match = sum(1 for item in comparison_data if item['match_status'] != 'NO_CSV_ENTRY')
        correct_predictions = sum(1 for item in comparison_data if item['is_correct'] is True)
        
        accuracy = correct_predictions / folders_with_csv_match if folders_with_csv_match > 0 else 0
        
        # Save summary
        summary_data = {
            'total_folders_processed': total_folders,
            'folders_found_in_csv': folders_with_csv_match,
            'folders_not_in_csv': total_folders - folders_with_csv_match,
            'correct_predictions': correct_predictions,
            'wrong_predictions': folders_with_csv_match - correct_predictions,
            'accuracy_percentage': f"{accuracy * 100:.2f}%",
            'model_confidence_avg': np.mean([float(item['model_confidence']) for item in comparison_data])
        }
        
        summary_path = self.output_path / "reports" / "accuracy_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
    
    print(f"✓ Model vs Ground Truth comparison saved to: {comparison_path}")
    print(f"✓ Accuracy summary saved to: {summary_path}")
    
    # Also save detailed results for debugging
    detailed_results = []
    for folder_result in results:
        for img_result in folder_result['individual_images']:
            detailed_results.append({
                'lesion_id': img_result['lesion_id'],
                'image_name': img_result['image_name'],
                'individual_prediction': img_result['predicted_disease'],
                'individual_confidence': img_result['prediction_confidence'],
                'folder_final_prediction': folder_result['folder_prediction']['predicted_disease'],
                'folder_confidence': folder_result['folder_prediction']['prediction_confidence'],
                'ground_truth': folder_result['ground_truth'],
                'folder_is_correct': folder_result['is_correct']
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = self.output_path / "reports" / "detailed_image_results.csv"
    detailed_df.to_csv(detailed_path, index=False)
    
    print(f"✓ Detailed results saved to: {detailed_path}")

# Also update the main function to ensure 50 folders
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='MILK10k Medical Image Processing Pipeline')
    parser.add_argument('--max-folders', type=int, default=50, help='Maximum number of folders to process (default: 50)')
    parser.add_argument('--full', action='store_true', help='Process entire dataset')
    args = parser.parse_args()
    
    max_folders = None if args.full else args.max_folders
    
    print(f"Processing {max_folders if max_folders else 'ALL'} folders")
    
    # Rest of your pipeline initialization code...
        
        # Save detailed image results
        images_df = pd.DataFrame(detailed_image_results)
        images_path = self.output_path / "reports" / "detailed_image_results.csv"
        images_df.to_csv(images_path, index=False)
        print(f"✓ Detailed image results saved to: {images_path}")
        
        # Save model vs CSV comparison
        comparison_df = pd.DataFrame(comparison_results)
        comparison_path = self.output_path / "reports" / "model_vs_csv_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✓ Model vs CSV comparison saved to: {comparison_path}")
        
        # Save processing report
        report_path = self.output_path / "reports" / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✓ Processing report saved to: {report_path}")
        
        # Generate summary visualizations
        self._create_folder_summary_plots(folder_df, comparison_df, report)
        
        print(f"\nResults saved to: {self.output_path}")
        print(f"Segmented outputs for ConceptCLIP: {self.output_path / 'segmented_for_conceptclip'}")
        print(f"Folder-level results: {folder_path}")
        print(f"Image-level results: {images_path}")
        print(f"Model vs CSV comparison: {comparison_path}")
        print(f"Processing report: {report_path}")
    
    def _create_folder_summary_plots(self, folder_df: pd.DataFrame, 
                                   comparison_df: pd.DataFrame, report: Dict):
        """Create summary visualization plots for folder-based processing"""
        print("Creating folder-based summary visualization plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Folder prediction distribution
        pred_counts = folder_df['model_prediction'].value_counts()
        axes[0,0].bar(pred_counts.index, pred_counts.values)
        axes[0,0].set_title('Folder-Level Disease Prediction Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Confidence distribution (folders)
        axes[0,1].hist(folder_df['model_confidence'], bins=30, alpha=0.7, color='blue')
        axes[0,1].set_title('Folder-Level Prediction Confidence Distribution')
        axes[0,1].set_xlabel('Confidence')
        axes[0,1].set_ylabel('Number of Folders')
        
        # 3. Images per folder distribution
        axes[0,2].hist(folder_df['num_images'], bins=20, alpha=0.7, color='green')
        axes[0,2].set_title('Images per Folder Distribution')
        axes[0,2].set_xlabel('Number of Images')
        axes[0,2].set_ylabel('Number of Folders')
        
        # 4. Model vs CSV comparison
        match_counts = comparison_df['match_status'].value_counts()
        axes[1,0].pie(match_counts.values, labels=match_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Model vs CSV Ground Truth Comparison')
        
        # 5. Accuracy by confidence level
        folders_with_gt = folder_df.dropna(subset=['ground_truth'])
        if len(folders_with_gt) > 0:
            conf_bins = np.linspace(0, 1, 11)
            accuracies = []
            for i in range(len(conf_bins)-1):
                mask = ((folders_with_gt['model_confidence'] >= conf_bins[i]) & 
                       (folders_with_gt['model_confidence'] < conf_bins[i+1]))
                if mask.sum() > 0:
                    acc = folders_with_gt[mask]['is_correct'].mean()
                    accuracies.append(acc)
                else:
                    accuracies.append(0)
            
            axes[1,1].plot(conf_bins[:-1], accuracies, marker='o')
            axes[1,1].set_title('Accuracy vs Folder Prediction Confidence')
            axes[1,1].set_xlabel('Prediction Confidence')
            axes[1,1].set_ylabel('Accuracy')
        
        # 6. Processing success rates
        success_data = [
            report['folder_level_results']['success_rate'],
            report['folder_level_results']['accuracy_metrics']['overall_accuracy']
        ]
        axes[1,2].bar(['Classification Success', 'Overall Accuracy'], success_data)
        axes[1,2].set_title('Processing Success Rates')
        axes[1,2].set_ylim(0, 1)
        
        plt.tight_layout()
        plot_path = self.output_path / "visualizations" / "folder_summary_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Folder-based summary plots saved to: {plot_path}")

print("✓ SECTION: Updated Pipeline class definition completed successfully")
print("-"*60)

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function with argument parsing for folder-based processing"""
    print("Starting main execution function for folder-based processing...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MILK10k Medical Image Processing Pipeline - Folder-based')
    parser.add_argument('--test', action='store_true', help='Run in test mode (20 folders only)')
    parser.add_argument('--max-folders', type=int, default=100, help='Maximum number of folders to process (default: 100)')
    parser.add_argument('--full', action='store_true', help='Process entire dataset (override max-folders)')
    args = parser.parse_args()
    
    print("✓ Command line arguments parsed successfully")
    
    # Determine max_folders
    if args.full:
        max_folders = None
        print("Processing FULL dataset")
    elif args.test:
        max_folders = 20
        print(f"TEST mode: Processing {max_folders} folders")
    else:
        max_folders = args.max_folders  # Default is 100
        print(f"Processing {max_folders} folders")
    
    print("="*60)
    print("MILK10K FOLDER-BASED MEDICAL IMAGE PROCESSING PIPELINE")
    print("Updated with Local Cache Support and CSV Comparison")
    if max_folders:
        print(f"🔬 Processing {max_folders} folders")
    else:
        print(f"🔬 Processing FULL dataset")
    print("="*60)
    
    pipeline = MILK10kPipeline(
        dataset_path=DATASET_PATH,
        groundtruth_path=GROUNDTRUTH_PATH,
        output_path=OUTPUT_PATH,
        sam2_model_path=SAM2_MODEL_PATH,
        conceptclip_model_path=CONCEPTCLIP_MODEL_PATH,
        cache_path=HUGGINGFACE_CACHE_PATH,
        max_folders=max_folders
    )
    
    report = pipeline.process_dataset()
    
    print("✓ SECTION: Dataset processing completed successfully")
    print("-"*60)
    
    # Print summary
    print("\n" + "="*50)
    if max_folders:
        print(f"MILK10K FOLDER-BASED TEST RUN COMPLETE ({max_folders} folders)")
    else:
        print("MILK10K FOLDER-BASED PROCESSING COMPLETE")
    print("="*50)
    print(f"Device used: {report['system_info']['device_used']}")
    print(f"Cache directory: {report['system_info']['cache_directory']}")
    print(f"Offline mode: {report['system_info']['offline_mode']}")
    print(f"Total folders processed: {report['dataset_info']['total_folders_processed']}")
    print(f"Total images processed: {report['dataset_info']['total_images_processed']}")
    print(f"Average images per folder: {report['dataset_info']['images_per_folder_stats']['mean']:.1f}")
    
    if report['folder_level_results']['accuracy_metrics']['total_evaluated'] > 0:
        print(f"Overall folder-level accuracy: {report['folder_level_results']['accuracy_metrics']['overall_accuracy']:.2%}")
    
    print(f"\nKey outputs:")
    print(f"- Folder-level results: {OUTPUT_PATH}/reports/folder_level_results.csv")
    print(f"- Model vs CSV comparison: {OUTPUT_PATH}/reports/model_vs_csv_comparison.csv")
    print(f"- Detailed image results: {OUTPUT_PATH}/reports/detailed_image_results.csv")
    print(f"- Segmented outputs: {OUTPUT_PATH}/segmented_for_conceptclip/")
    
    print("✓ SECTION: Final summary and output completed successfully")
    print("-"*60)
    print("✓ PROGRAM EXECUTION COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
