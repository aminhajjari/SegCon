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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from skimage import filters, morphology, measure
import warnings
import argparse
import sys
import logging
import time
from fuzzywuzzy import process
from scipy import ndimage
warnings.filterwarnings('ignore')

# Set up Python path for ConceptModel and SAM2 imports
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2')

from ConceptModel.modeling_conceptclip import ConceptCLIP
from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configuration
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_GroundTruth.csv"
MASKS_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Masks"
OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs"
SAM2_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2"
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"

# GPU Setup
def setup_gpu_environment():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("GPU ENVIRONMENT SETUP")
    logger.info("=" * 50)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB)")
        device = f"cuda:{torch.cuda.current_device()}"
        logger.info(f"Using device: {device}")
        try:
            test_tensor = torch.randn(10, 10).to(device)
            logger.info("✅ GPU allocation test successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"❌ GPU allocation test failed: {e}")
            device = "cpu"
    else:
        logger.warning("⚠️ CUDA not available. Using CPU.")
        device = "cpu"
        logger.info(f"SLURM_GPUS_ON_NODE: {os.environ.get('SLURM_GPUS_ON_NODE', 'Not set')}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    logger.info("=" * 50)
    return device

# SAM2 Loading
def load_local_sam2_model(model_path: str, device: str):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading SAM2 from local path: {model_path}")
    model_cfg = os.path.join(model_path, "sam2_configs", "sam2_hiera_l.yaml")
    sam2_checkpoint = os.path.join(model_path, "checkpoints", "sam2_hiera_large.pt")
    
    if not os.path.exists(model_cfg) or not os.path.exists(sam2_checkpoint):
        logger.error(f"Missing SAM2 files: config={model_cfg}, checkpoint={sam2_checkpoint}")
        raise FileNotFoundError("SAM2 model files not found")
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    logger.info(f"✅ SAM2 loaded successfully on {device}")
    return predictor

# Offline Setup
def setup_offline_environment(cache_path: str):
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("OFFLINE ENVIRONMENT SETUP")
    logger.info("=" * 50)
    
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    logger.info("✅ Offline mode enabled")
    logger.info(f"✅ Cache directory set to: {cache_path}")
    
    cache_path_obj = Path(cache_path)
    if cache_path_obj.exists():
        logger.info("✅ Cache directory exists")
        cached_models = list(cache_path_obj.glob("models--*"))
        logger.info(f"✅ Found {len(cached_models)} cached models:")
        for model in cached_models:
            logger.info(f"   - {model.name}")
    else:
        logger.error(f"❌ Cache directory does not exist: {cache_path}")
        raise FileNotFoundError("Cache directory not found")
    
    logger.info("=" * 50)

# ConceptCLIP Loading
def load_local_conceptclip_models(model_path: str, cache_path: str, device: str):
    logger = logging.getLogger(__name__)
    setup_offline_environment(cache_path)
    logger.info(f"Loading ConceptCLIP from local path: {model_path}")
    model = ConceptCLIP.from_pretrained(
        model_path,
        local_files_only=True,
        cache_dir=cache_path
    )
    processor = ConceptCLIPProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        cache_dir=cache_path
    )
    model = model.to(device)
    model.eval()
    logger.info(f"✅ ConceptCLIP loaded successfully on {device}")
    return model, processor

# MILK10k Domain
@dataclass
class MedicalDomain:
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

# Main Pipeline
class MILK10kPipeline:
    def __init__(self, dataset_path: str, groundtruth_path: str, output_path: str, 
                 sam2_model_path: str = None, conceptclip_model_path: str = None,
                 cache_path: str = None, max_folders: int = None):
        
        # FIX: Assign self.output_path first
        self.output_path = Path(output_path)
        
        # Ensure the logs directory exists before setting up the file handler
        log_dir = self.output_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.output_path / "logs" / "pipeline.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        torch.manual_seed(42)  # For reproducibility
        self.logger.info("Initializing MILK10k Pipeline...")
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.masks_path = MASKS_PATH
        self.sam2_model_path = sam2_model_path or SAM2_MODEL_PATH
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.cache_path = cache_path or HUGGINGFACE_CACHE_PATH
        self.domain = MILK10K_DOMAIN
        self.max_folders = max_folders
        
        # The other directory creations can remain, but it's good practice to ensure they exist as well
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "segmented").mkdir(exist_ok=True)
        (self.output_path / "segmented_for_conceptclip").mkdir(exist_ok=True)
        (self.output_path / "classifications").mkdir(exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        
        self.device = setup_gpu_environment()
        self._load_models()
        self._load_ground_truth()
        self.logger.info("✓ SECTION: Pipeline initialization completed successfully")
        
    def _load_models(self):
        self.logger.info("Loading models...")
        self.sam_predictor = load_local_sam2_model(self.sam2_model_path, self.device)
        self.conceptclip_model, self.conceptclip_processor = load_local_conceptclip_models(
            self.conceptclip_model_path, self.cache_path, self.device
        )
        self.logger.info("✓ SECTION: Model loading completed successfully")
        
    def _load_ground_truth(self):
        self.logger.info("Loading ground truth data...")
        if os.path.exists(self.groundtruth_path):
            self.ground_truth = pd.read_csv(self.groundtruth_path)
            self.logger.info(f"Loaded ground truth: {len(self.ground_truth)} samples")
            expected_columns = ['lesion_id'] + list(self.domain.label_mappings.keys())
            if not all(col in self.ground_truth.columns for col in expected_columns):
                self.logger.warning(f"Missing expected columns in CSV. Found: {list(self.ground_truth.columns)}")
            
            self.gt_lookup = {}
            for _, row in self.ground_truth.iterrows():
                lesion_id = str(row['lesion_id']).strip().lower()
                for col in self.domain.label_mappings.keys():
                    if col in row and float(row[col]) == 1.0:
                        self.gt_lookup[lesion_id] = self.domain.label_mappings[col]
                        break
            self.logger.info(f"✓ Created lookup table for {len(self.gt_lookup)} lesions")
        else:
            self.logger.error(f"Ground truth file not found: {self.groundtruth_path}")
            self.ground_truth = None
            self.gt_lookup = {}
        self.logger.info("✓ SECTION: Ground truth loading completed successfully")
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
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
            self.logger.error(f"Error loading {image_path}: {e}")
            return None
    
    def _load_dicom(self, image_path: Path) -> np.ndarray:
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array.astype(np.float32)
        image = self._normalize_image(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    
    def _load_nifti(self, image_path: Path) -> np.ndarray:
        nii_img = nib.load(image_path)
        image = nii_img.get_fdata()
        if len(image.shape) == 3:
            mid_slice = image.shape[2] // 2
            image = image[:, :, mid_slice]
        image = self._normalize_image(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    
    def _load_standard_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.domain.preprocessing_params.get('enhance_contrast', False):
            image = self._enhance_contrast(image)
        return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
        return (image * 255).astype(np.uint8)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _find_automatic_prompts(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Dark spot detection
        dark_thresh = np.percentile(gray, 25)
        dark_mask = gray < dark_thresh
        dark_mask = morphology.binary_closing(dark_mask, morphology.disk(5))
        dark_mask = morphology.binary_opening(dark_mask, morphology.disk(3))

        # Find lesion centroids
        labeled = measure.label(dark_mask)
        regions = measure.regionprops(labeled)
        min_area = (h * w) * 0.005
        max_area = (h * w) * 0.3
        valid_regions = [r for r in regions if min_area < r.area < max_area]
        valid_regions = sorted(valid_regions, key=lambda x: x.area, reverse=True)[:3]

        positive_points = []
        for region in valid_regions:
            y, x = region.centroid
            positive_points.append([int(x), int(y)])

        if not positive_points:
            positive_points.append([w // 2, h // 2])

        negative_points = [[20, 20], [w-20, h-20]]

        all_points = positive_points + negative_points
        all_labels = [1] * len(positive_points) + [0] * len(negative_points)
        return np.array(all_points), np.array(all_labels)
    
    def segment_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        h, w = image.shape[:2]
        point_coords, point_labels = self._find_automatic_prompts(image)
        masks_and_scores = []
        sam2_success = False
        
        try:
            with torch.inference_mode():
                self.sam_predictor.set_image(image)
                try:
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
                    for mask, score in zip(masks, scores):
                        if score > 0.5:
                            processed_mask = self._post_process_mask(mask, image)
                            masks_and_scores.append((processed_mask, float(score)))
                    sam2_success = True
                except Exception as e:
                    self.logger.error(f"Strategy 1 failed: {e}")
                
                if not sam2_success:
                    positive_only = point_coords[point_labels == 1]
                    if len(positive_only) > 0:
                        masks, scores, _ = self.sam_predictor.predict(
                            point_coords=positive_only,
                            point_labels=np.ones(len(positive_only)),
                            multimask_output=True
                        )
                        for mask, score in zip(masks, scores):
                            if score > 0.5:
                                processed_mask = self._post_process_mask(mask, image)
                                masks_and_scores.append((processed_mask, float(score)))
                        sam2_success = True
                
                if not sam2_success:
                    center_point = np.array([[w // 2, h // 2]])
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=center_point,
                        point_labels=np.array([1]),
                        multimask_output=True
                    )
                    for mask, score in zip(masks, scores):
                        if score > 0.5:
                            processed_mask = self._post_process_mask(mask, image)
                            masks_and_scores.append((processed_mask, float(score)))
                    sam2_success = True
        except Exception as e:
            self.logger.error(f"SAM2 segmentation error: {e}")
        
        if not masks_and_scores:
            self.logger.error("All SAM2 strategies failed; no valid masks produced")
            raise RuntimeError("SAM2 segmentation failed")
        
        self.logger.info(f"SAM2 status: Success")
        torch.cuda.empty_cache()
        return masks_and_scores
    
    def _post_process_mask(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        mask = cv2.medianBlur(mask, 5)
        mask_filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask)
            cv2.fillPoly(clean_mask, [largest_contour], 1)
            return clean_mask
        return mask
    
    def evaluate_segmentation(self, predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Dict:
        intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
        union = np.logical_or(predicted_mask, ground_truth_mask).sum()
        dice = 2 * intersection / (predicted_mask.sum() + ground_truth_mask.sum() + 1e-5)
        iou = intersection / (union + 1e-5)
        return {'dice': float(dice), 'iou': float(iou)}
    
    def create_segmented_outputs(self, image: np.ndarray, masks_and_scores: List[Tuple[np.ndarray, float]]) -> Dict[str, List[np.ndarray]]:
        outputs = defaultdict(list)
        h, w = image.shape[:2]
        
        for mask_idx, (mask, score) in enumerate(masks_and_scores):
            if mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)
            
            overlay_color = (255, 100, 100)
            alpha = 0.4
            overlay = image.copy()
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 1] = overlay_color
            colored_overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
            outputs['colored_overlay'].append(colored_overlay)
            
            contour_image = image.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
            outputs['contour'].append(contour_image)
            
            coords = np.where(mask == 1)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                lesion_h, lesion_w = y_max - y_min, x_max - x_min
                padding_y = max(20, int(lesion_h * 0.2))
                padding_x = max(20, int(lesion_w * 0.2))
                y_min = max(0, y_min - padding_y)
                y_max = min(h, y_max + padding_y)
                x_min = max(0, x_min - padding_x)
                x_max = min(w, x_max + padding_x)
                cropped = colored_overlay[y_min:y_max, x_min:x_max]
                if cropped.size > 0:
                    outputs['cropped'].append(cropped)
            
            masked_only = np.zeros_like(image)
            masked_only[mask == 1] = image[mask == 1]
            outputs['masked_only'].append(masked_only)
            
            comparison = np.hstack([image, colored_overlay])
            outputs['side_by_side'].append(comparison)
            
            if 'cropped' in outputs:
                try:
                    cropped_enhanced = self._enhance_contrast(cropped)
                    outputs['cropped_enhanced'].append(cropped_enhanced)
                except:
                    pass
        
        return outputs
    
    def classify_segmented_image(self, segmented_outputs: Dict[str, List[np.ndarray]]) -> Dict:
        try:
            batch_images = []
            batch_types = []
            for output_type, seg_images in segmented_outputs.items():
                for i, seg_image in enumerate(seg_images):
                    if seg_image is not None and seg_image.size > 0:
                        if min(seg_image.shape[:2]) < 64:
                            seg_image = cv2.resize(seg_image, (224, 224))
                        batch_images.append(Image.fromarray(seg_image.astype(np.uint8)))
                        batch_types.append(f"{output_type}_{i}")
            
            if batch_images:
                inputs = self.conceptclip_processor(
                    images=batch_images,
                    text=self.domain.text_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    outputs = self.conceptclip_model(**inputs)
                    results = {}
                    for i, output_type in enumerate(batch_types):
                        logits = outputs['logit_scale'] * outputs['image_features'][i] @ outputs['text_features'].t()
                        probs = logits.softmax(dim=-1).cpu().numpy()[0]
                        disease_names = [prompt.split(' showing ')[-1] for prompt in self.domain.text_prompts]
                        results[output_type] = {disease_names[j]: float(probs[j]) for j in range(len(disease_names))}
                    return self._ensemble_predictions(results)
            return {}
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return {}
    
    def _ensemble_predictions(self, results: Dict) -> Dict:
        if not results:
            return {}
        disease_names = list(next(iter(results.values())).keys())
        max_probs = {disease: 0.0 for disease in disease_names}
        
        for output_type, probs in results.items():
            for disease in disease_names:
                if disease in probs:
                    max_probs[disease] = max(max_probs[disease], probs[disease])
        
        total = sum(max_probs.values()) + 1e-5
        return {disease: prob / total for disease, prob in max_probs.items()}
    
    def process_folder(self, folder_path: Path) -> Dict:
        self.logger.info(f"Starting processing for folder {folder_path.name}")
        folder_results = []
        lesion_id = folder_path.name.lower()
        
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
        
        if not image_files:
            self.logger.warning(f"No images found in folder: {lesion_id}")
            return None
        
        self.logger.info(f"Processing folder {lesion_id} with {len(image_files)} images")
        error_count = 0
        max_errors = max(1, len(image_files) // 2)
        
        for img_idx, img_path in enumerate(image_files):
            try:
                image = self.preprocess_image(img_path)
                if image is None:
                    error_count += 1
                    if error_count > max_errors:
                        self.logger.warning(f"Too many errors ({error_count}/{max_errors}) in folder {lesion_id}, skipping")
                        return None
                    continue
                
                masks_and_scores = self.segment_image(image)
                segmented_outputs = self.create_segmented_outputs(image, masks_and_scores)
                
                img_name = img_path.stem
                conceptclip_dir = self.output_path / "segmented_for_conceptclip" / lesion_id / img_name
                conceptclip_dir.mkdir(exist_ok=True, parents=True)
                
                for output_type, seg_images in segmented_outputs.items():
                    for i, seg_image in enumerate(seg_images):
                        if seg_image is not None and seg_image.size > 0:
                            output_path = conceptclip_dir / f"{output_type}_{i}.png"
                            cv2.imwrite(str(output_path), cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
                
                classification_probs = self.classify_segmented_image(segmented_outputs)
                predicted_disease = max(classification_probs, key=classification_probs.get) if classification_probs else "unknown"
                prediction_confidence = classification_probs.get(predicted_disease, 0.0)
                seg_confidence = max(score for _, score in masks_and_scores) if masks_and_scores else 0.0
                
                # Comment out if no ground-truth masks
                seg_metrics = {}
                gt_mask_path = Path(self.masks_path) / lesion_id / f"{img_name}.png"
                if gt_mask_path.exists():
                    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None:
                        best_mask = max(masks_and_scores, key=lambda x: x[1])[0] if masks_and_scores else np.zeros_like(image[..., 0])
                        seg_metrics = self.evaluate_segmentation(best_mask, gt_mask > 0)
                
                folder_results.append({
                    'lesion_id': lesion_id,
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'image_index_in_folder': img_idx,
                    'predicted_disease': predicted_disease,
                    'prediction_confidence': prediction_confidence,
                    'segmentation_confidence': seg_confidence,
                    'segmentation_metrics': seg_metrics,
                    'classification_probabilities': classification_probs,
                    'segmented_outputs_dir': str(conceptclip_dir)
                })
            except Exception as e:
                self.logger.error(f"Error processing image {img_path}: {e}")
                error_count += 1
                if error_count > max_errors:
                    self.logger.warning(f"Too many errors ({error_count}/{max_errors}) in folder {lesion_id}, skipping")
                    return None
                continue
        
        if not folder_results:
            return None
        
        folder_prediction = self._aggregate_folder_predictions(folder_results)
        ground_truth = self._get_ground_truth(lesion_id)
        is_correct = ground_truth == folder_prediction['predicted_disease'] if ground_truth else None
        
        result = {
            'lesion_id': lesion_id,
            'num_images': len(folder_results),
            'individual_images': folder_results,
            'folder_prediction': folder_prediction,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'csv_match_found': ground_truth is not None
        }
        self.logger.info(f"Completed processing for folder {folder_path.name}")
        torch.cuda.empty_cache()
        return result
    
    def _get_ground_truth(self, lesion_id: str) -> Optional[str]:
        lesion_id_lower = lesion_id.lower()
        if lesion_id_lower in self.gt_lookup:
            return self.gt_lookup[lesion_id_lower]
        if self.gt_lookup:
            closest_match, score = process.extractOne(lesion_id_lower, self.gt_lookup.keys())
            if score > 90:
                self.logger.info(f"Fuzzy matched {lesion_id} to {closest_match} (score: {score})")
                return self.gt_lookup[closest_match]
        return None
    
    def _aggregate_folder_predictions(self, folder_results: List[Dict]) -> Dict:
        if not folder_results:
            return {'predicted_disease': 'unknown', 'prediction_confidence': 0.0}
        all_diseases = self.domain.class_names
        weighted_probs = {disease: 0.0 for disease in all_diseases}
        total_weights = 0.0
        
        for result in folder_results:
            probs = result.get('classification_probabilities', {})
            confidence = result.get('prediction_confidence', 0.0)
            weight = max(0.1, confidence)
            for disease in all_diseases:
                if disease in probs:
                    weighted_probs[disease] += weight * probs[disease]
            total_weights += weight
        
        if total_weights > 0:
            for disease in weighted_probs:
                weighted_probs[disease] /= total_weights
        
        predicted_disease = max(weighted_probs, key=weighted_probs.get)
        prediction_confidence = weighted_probs[predicted_disease]
        predictions = [result['predicted_disease'] for result in folder_results]
        prediction_counts = Counter(predictions)
        majority_prediction = prediction_counts.most_common(1)[0][0]
        
        return {
            'predicted_disease': predicted_disease,
            'prediction_confidence': prediction_confidence,
            'weighted_probabilities': weighted_probs,
            'majority_vote_prediction': majority_prediction,
            'prediction_counts': dict(prediction_counts),
            'aggregation_method': 'confidence_weighted_average',
            'total_weight': total_weights
        }
    
    def _generate_folder_based_report(self, results: List[Dict], accuracy: float, 
                                    total_with_gt: int, total_images: int) -> Dict:
        self.logger.info("Generating folder-based report...")
        total_folders = len(results)
        folder_predictions = [r['folder_prediction']['predicted_disease'] for r in results]
        prediction_counts = Counter(folder_predictions)
        
        labels = self.domain.class_names
        confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
        f1_scores = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in labels}
        
        for result in results:
            pred = result['folder_prediction']['predicted_disease']
            gt = result['ground_truth']
            if gt and pred in labels and gt in labels:
                pred_idx = labels.index(pred)
                gt_idx = labels.index(gt)
                confusion_matrix[gt_idx, pred_idx] += 1
                if pred == gt:
                    f1_scores[pred]['tp'] += 1
                else:
                    f1_scores[pred]['fp'] += 1
                    f1_scores[gt]['fn'] += 1
        
        f1_results = {}
        for label in labels:
            tp = f1_scores[label]['tp']
            fp = f1_scores[label]['fp']
            fn = f1_scores[label]['fn']
            precision = tp / (tp + fp + 1e-5)
            recall = tp / (tp + fn + 1e-5)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
            f1_results[label] = float(f1)
        
        # Comment out if no ground-truth masks
        segmentation_metrics = []
        for result in results:
            for img_result in result['individual_images']:
                if img_result.get('segmentation_metrics'):
                    segmentation_metrics.append(img_result['segmentation_metrics'])
        
        device_used = results[0]['device_used'] if results else "unknown"
        
        report = {
            'system_info': {
                'device_used': device_used,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'offline_mode': os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1",
                'cache_directory': self.cache_path
            },
            'dataset_info': {
                'total_folders_processed': total_folders,
                'folders_with_ground_truth': total_with_gt,
                'total_images_processed': total_images
            },
            'classification_metrics': {
                'overall_accuracy': accuracy,
                'correct_predictions': sum(1 for r in results if r['is_correct']),
                'total_evaluated': total_with_gt,
                'per_class_f1': f1_results,
                'confusion_matrix': confusion_matrix.tolist()
            },
            'segmentation_metrics': {  # Comment out if no masks
                'mean_dice': float(np.mean([m['dice'] for m in segmentation_metrics])) if segmentation_metrics else 0.0,
                'mean_iou': float(np.mean([m['iou'] for m in segmentation_metrics])) if segmentation_metrics else 0.0,
                'num_evaluated': len(segmentation_metrics)
            },
            'predictions': {
                'distribution': dict(prediction_counts),
                'most_common': prediction_counts.most_common(5)
            }
        }
        
        self._create_folder_summary_plots(report)
        return report
    
    def _save_folder_results(self, results: List[Dict], report: Dict):
        self.logger.info("Saving results to CSV...")
        comparison_data = []
        
        for folder_result in results:
            lesion_id = folder_result['lesion_id']
            model_prediction = folder_result['folder_prediction']['predicted_disease']
            model_confidence = folder_result['folder_prediction']['prediction_confidence']
            ground_truth = folder_result['ground_truth']
            reverse_mapping = {v: k for k, v in self.domain.label_mappings.items()}
            model_prediction_code = reverse_mapping.get(model_prediction, model_prediction)
            ground_truth_code = reverse_mapping.get(ground_truth, ground_truth) if ground_truth else None
            is_correct = (model_prediction == ground_truth) if ground_truth else None
            
            comparison_data.append({
                'lesion_id': lesion_id,
                'ground_truth_disease': ground_truth if ground_truth else 'NOT_FOUND_IN_CSV',
                'predicted_disease': model_prediction,
                'confidence': f"{model_confidence:.4f}",
                'is_correct': is_correct,
                'match_status': 'CORRECT' if is_correct else ('WRONG' if ground_truth else 'NO_CSV_ENTRY')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = self.output_path / "reports" / "results_summary.csv"
        
        summary_lines = [
            f"# Total Folders Processed: {report['dataset_info']['total_folders_processed']}",
            f"# Folders with Ground Truth: {report['dataset_info']['folders_with_ground_truth']}",
            f"# Correct Predictions: {report['classification_metrics']['correct_predictions']}",
            f"# Accuracy: {report['classification_metrics']['overall_accuracy']:.2%}",
            f"# Per-Class F1-Scores: {json.dumps(report['classification_metrics']['per_class_f1'], default=str)}",
            f"# Confusion Matrix: {json.dumps(report['classification_metrics']['confusion_matrix'], default=str)}",
            f"# Segmentation Metrics: Mean Dice = {report['segmentation_metrics']['mean_dice']:.4f}, Mean IoU = {report['segmentation_metrics']['mean_iou']:.4f}, Num Evaluated = {report['segmentation_metrics']['num_evaluated']}"
        ]
        with open(comparison_path, 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')
            comparison_df.to_csv(f, index=False)
        
        self.logger.info(f"✓ Results saved to: {comparison_path}")
    
    def _create_folder_summary_plots(self, report: Dict):
        self.logger.info("Creating Chart.js prediction distribution config...")
        pred_dist = report['predictions']['distribution']
        labels = list(pred_dist.keys())
        values = list(pred_dist.values())
        colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", 
                  "#FF9F40", "#FF6666", "#66CCCC", "#CC99CC", "#99CC99", "#6666FF"]
        
        pred_plot = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Folder-Level Predictions",
                    "data": values,
                    "backgroundColor": colors[:len(labels)],
                    "borderColor": colors[:len(labels)],
                    "borderWidth": 1
                }]
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Number of Folders"}
                    },
                    "x": {
                        "title": {"display": True, "text": "Disease"}
                    }
                },
                "plugins": {
                    "title": {"display": True, "text": "Folder-Level Disease Prediction Distribution"},
                    "legend": {"display": False}
                }
            }
        }
        
        plot_path = self.output_path / "visualizations" / "prediction_distribution.json"
        with open(plot_path, 'w') as f:
            json.dump(pred_plot, f, indent=2)
        self.logger.info(f"✓ Chart.js config saved to: {plot_path}")
    
    def process_dataset(self) -> Dict:
        start_time = time.time()
        self.logger.info("Starting MILK10k dataset processing...")
        lesion_folders = [f for f in self.dataset_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
        self.logger.info(f"Found {len(lesion_folders)} lesion folders in dataset")
        lesion_folders = sorted(lesion_folders, key=lambda x: x.name)
        
        max_failed_folders = max(1, len(lesion_folders) // 10)
        failed_folders = 0
        
        if self.max_folders:
            original_count = len(lesion_folders)
            lesion_folders = lesion_folders[:self.max_folders]
            self.logger.info(f"Processing {len(lesion_folders)} folders out of {original_count} total")
        else:
            self.logger.info(f"Processing all {len(lesion_folders)} folders")
        
        results = []
        correct_predictions = 0
        total_with_gt = 0
        csv_matches_found = 0
        total_images_processed = 0
        
        desc = f"Processing {'Limited' if self.max_folders else 'Full'} MILK10k folders"
        for folder_idx, folder_path in enumerate(tqdm(lesion_folders, desc=desc)):
            try:
                folder_result = self.process_folder(folder_path)
                if folder_result is None:
                    failed_folders += 1
                    if failed_folders > max_failed_folders:
                        self.logger.warning(f"Too many failed folders ({failed_folders}/{max_failed_folders}), stopping early")
                        break
                    continue
                total_images_processed += folder_result['num_images']
                if folder_result['csv_match_found']:
                    csv_matches_found += 1
                    total_with_gt += 1
                    if folder_result['is_correct']:
                        correct_predictions += 1
                folder_result['folder_index'] = folder_idx
                folder_result['processing_mode'] = 'limited' if self.max_folders else 'full'
                folder_result['device_used'] = self.device
                results.append(folder_result)
                
                lesion_id = folder_result['lesion_id']
                prediction = folder_result['folder_prediction']['predicted_disease']
                confidence = folder_result['folder_prediction']['prediction_confidence']
                status = "✓" if folder_result['is_correct'] else ("✗" if folder_result['ground_truth'] else "-")
                csv_status = "CSV✓" if folder_result['csv_match_found'] else "CSV✗"
                self.logger.info(f"{status} {csv_status} [{folder_idx+1:3d}] {lesion_id}: {prediction} ({confidence:.2%}) [{folder_result['num_images']} imgs]")
            except Exception as e:
                self.logger.error(f"Error processing folder {folder_path}: {e}")
                failed_folders += 1
                if failed_folders > max_failed_folders:
                    self.logger.warning(f"Too many failed folders ({failed_folders}/{max_failed_folders}), stopping early")
                    break
                continue
        
        accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
        report = self._generate_folder_based_report(results, accuracy, total_with_gt, total_images_processed)
        self._save_folder_results(results, report)
        self.logger.info(f"Processing complete. Total time: {time.time() - start_time:.2f}s")
        return report

def main():
    parser = argparse.ArgumentParser(description='MILK10k Medical Image Processing Pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode (20 folders)')
    parser.add_argument('--max-folders', type=int, default=50, help='Maximum number of folders to process')
    parser.add_argument('--full', action='store_true', help='Process entire dataset')
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logger.info("✓ Command line arguments parsed successfully")
    if args.full:
        max_folders = None
        logger.info("Processing FULL dataset")
    elif args.test:
        max_folders = 20
        logger.info(f"TEST mode: Processing {max_folders} folders")
    else:
        max_folders = args.max_folders
        logger.info(f"Processing {max_folders} folders")
    
    logger.info("="*60)
    logger.info("MILK10K FOLDER-BASED MEDICAL IMAGE PROCESSING PIPELINE")
    logger.info("Updated with Simplified Metrics and Improvements")
    logger.info(f"🔬 Processing {'FULL dataset' if max_folders is None else f'{max_folders} folders'}")
    logger.info("="*60)
    
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
    
    logger.info("✓ SECTION: Dataset processing completed successfully")
    logger.info("-"*60)
    
    logger.info("\n" + "="*50)
    logger.info(f"MILK10K PROCESSING COMPLETE ({'FULL' if max_folders is None else f'{max_folders} folders'})")
    logger.info("="*50)
    logger.info(f"Device used: {report['system_info']['device_used']}")
    logger.info(f"Cache directory: {report['system_info']['cache_directory']}")
    logger.info(f"Offline mode: {report['system_info']['offline_mode']}")
    logger.info(f"Total folders processed: {report['dataset_info']['total_folders_processed']}")
    logger.info(f"Total images processed: {report['dataset_info']['total_images_processed']}")
    
    if report['classification_metrics']['total_evaluated'] > 0:
        logger.info(f"Classification accuracy: {report['classification_metrics']['overall_accuracy']:.2%}")
    if report['segmentation_metrics']['num_evaluated'] > 0:
        logger.info(f"Segmentation metrics: Mean Dice = {report['segmentation_metrics']['mean_dice']:.4f}, Mean IoU = {report['segmentation_metrics']['mean_iou']:.4f}")
    
    logger.info(f"\nKey outputs:")
    logger.info(f"- Results CSV: {OUTPUT_PATH}/reports/results_summary.csv")
    logger.info(f"- Visualization: {OUTPUT_PATH}/visualizations/prediction_distribution.json")
    logger.info(f"- Segmented outputs: {OUTPUT_PATH}/segmented_for_conceptclip/")
    logger.info(f"- Log file: {OUTPUT_PATH}/logs/pipeline.log")
    
    logger.info("✓ SECTION: Final summary and output completed successfully")
    logger.info("✓ PROGRAM EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("="*60)

if __name__ == "__main__":
    main()
