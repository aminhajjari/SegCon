# MILK10k Medical Image Segmentation and Classification Pipeline
# Final version with ConceptCLIP direct imports and error fixes

import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
import nibabel as nib
import SimpleITK as sitk
from sam2.sam2_image_predictor import SAM2ImagePredictor
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
warnings.filterwarnings('ignore')


import warnings
warnings.filterwarnings('ignore')

# Set up Python path for ConceptModel imports
import sys
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')

# Import local ConceptCLIP modules directly
# NEW (correct):
from ConceptModel.modeling_conceptclip import ConceptCLIP
from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor  # Check this class name too

# ==================== CONFIGURATION ====================

# Your dataset paths (Narval specific)
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/groundtruth.csv"
OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/outputs"

# Local model paths
SAM2_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2"
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"

# ==================== LOCAL MODEL LOADING ====================

def load_local_conceptclip_models(model_path: str, device: str):
    """Load local ConceptCLIP models using direct imports"""
    try:
        print(f"Loading ConceptCLIP from local path: {model_path}")
        
        # Load model and processor using correct class names
        # NEW (correct):
        model = ConceptCLIP.from_pretrained(model_path)
        processor = ConceptCLIPProcessor.from_pretrained(model_path)
        
        # Move to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"ConceptCLIP loaded successfully on {device}")
        return model, processor
        
    except Exception as e:
        print(f"Error loading local ConceptCLIP: {e}")
        print("Please check your ConceptCLIP model path and imports")
        raise e

def load_local_sam2_model(model_path: str):
    """Load local SAM2 model (already installed in editable mode)"""
    try:
        print("Loading SAM2 model (installed in editable mode)...")
        
        # Since SAM2 is installed in editable mode, we can use it directly
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        
        print("SAM2 loaded successfully from installed package")
        return predictor
        
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        print("Please check your SAM2 installation in the virtual environment")
        raise e

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

# MILK10k Medical Domain Configuration
MILK10K_DOMAIN = MedicalDomain(
    name="milk10k",
    image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.dcm', '.dicom'],
    text_prompts=[
        'a medical image showing normal tissue',
        'a medical image showing abnormal pathology',
        'a medical image showing inflammatory lesion',
        'a medical image showing neoplastic lesion',
        'a medical image showing degenerative changes',
        'a medical image showing infectious disease',
        'a medical image showing vascular pathology',
        'a medical image showing metabolic disorder',
        'a medical image showing congenital abnormality',
        'a medical image showing traumatic injury'
    ],
    label_mappings={
        'NORMAL': 'normal tissue',
        'ABNORMAL': 'abnormal pathology',
        'INFLAMMATORY': 'inflammatory lesion',
        'NEOPLASTIC': 'neoplastic lesion',
        'DEGENERATIVE': 'degenerative changes',
        'INFECTIOUS': 'infectious disease',
        'VASCULAR': 'vascular pathology',
        'METABOLIC': 'metabolic disorder',
        'CONGENITAL': 'congenital abnormality',
        'TRAUMATIC': 'traumatic injury'
    },
    preprocessing_params={'normalize': True, 'enhance_contrast': True},
    segmentation_strategy='adaptive'
)

# ==================== MAIN PIPELINE CLASS ====================

class MILK10kPipeline:
    """MILK10k segmentation and classification pipeline"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, output_path: str, 
                 sam2_model_path: str = None, conceptclip_model_path: str = None):
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.output_path = Path(output_path)
        self.sam2_model_path = sam2_model_path or SAM2_MODEL_PATH
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.domain = MILK10K_DOMAIN
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "segmented").mkdir(exist_ok=True)
        (self.output_path / "segmented_for_conceptclip").mkdir(exist_ok=True)  # Key directory
        (self.output_path / "classifications").mkdir(exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing MILK10k pipeline on {self.device}")
        
        # Load models
        self._load_models()
        
        # Load ground truth
        self._load_ground_truth()
        
    def _load_models(self):
        """Load local SAM2 and ConceptCLIP models"""
        
        # Load local SAM2
        self.sam_predictor = load_local_sam2_model(self.sam2_model_path)
        
        # Load local ConceptCLIP
        self.conceptclip_model, self.conceptclip_processor = load_local_conceptclip_models(
            self.conceptclip_model_path, self.device
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
        """Segment image using SAM2 with adaptive strategy"""
        h, w = image.shape[:2]
        
        # Adaptive segmentation strategy
        center_points = self._find_roi_points(image)
        point_labels = np.ones(len(center_points))
        
        # SAM2 segmentation with proper error handling
        try:
            with torch.inference_mode():
                # Use mixed precision for better performance on V100
                if torch.cuda.is_available():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        self.sam_predictor.set_image(image)
                        masks, scores, _ = self.sam_predictor.predict(
                            point_coords=center_points,
                            point_labels=point_labels,
                            box=None,
                            multimask_output=True
                        )
                else:
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
    
    def create_segmented_outputs(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Create multiple segmented outputs for ConceptCLIP"""
        outputs = {}
        
        # 1. Original image with colored overlay
        color = (255, 0, 0)  # Red highlight
        overlay = image.copy()
        overlay[mask == 1] = color
        colored_overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        outputs['colored_overlay'] = colored_overlay
        
        # 2. Contour highlighting
        contour_image = image.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
        outputs['contour'] = contour_image
        
        # 3. Cropped to bounding box
        coords = np.where(mask == 1)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Add padding
            h, w = image.shape[:2]
            padding = 20
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            
            cropped = image[y_min:y_max, x_min:x_max]
            outputs['cropped'] = cropped
        
        # 4. Masked region only (with black background)
        masked_only = image.copy()
        masked_only[mask == 0] = [0, 0, 0]
        outputs['masked_only'] = masked_only
        
        # 5. Side-by-side comparison
        if 'cropped' in outputs:
            # Resize cropped to match original height for side-by-side
            h_orig = image.shape[0]
            cropped_resized = cv2.resize(outputs['cropped'], 
                                       (int(outputs['cropped'].shape[1] * h_orig / outputs['cropped'].shape[0]), h_orig))
            side_by_side = np.hstack([image, cropped_resized])
            outputs['side_by_side'] = side_by_side
        
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
        """Process entire MILK10k dataset"""
        print("Starting MILK10k dataset processing...")
        
        # Find all images
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(self.dataset_path.rglob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images in dataset")
        
        results = []
        format_counter = Counter()
        correct_predictions = 0
        total_with_gt = 0
        
        for img_path in tqdm(image_files, desc="Processing MILK10k images"):
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
                    'classification_probabilities': classification_probs
                }
                
                results.append(result)
                
                # Progress indicator
                status = "✓" if result['correct'] else ("✗" if ground_truth else "-")
                print(f"{status} {img_name}: {predicted_disease} ({prediction_confidence:.2%})")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate accuracy
        accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
        
        # Generate report
        report = self._generate_comprehensive_report(results, format_counter, accuracy, total_with_gt)
        
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
        
        report = {
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
    """Main execution function"""
    
    # Initialize pipeline with local models
    pipeline = MILK10kPipeline(
        dataset_path=DATASET_PATH,
        groundtruth_path=GROUNDTRUTH_PATH,
        output_path=OUTPUT_PATH,
        sam2_model_path=SAM2_MODEL_PATH,
        conceptclip_model_path=CONCEPTCLIP_MODEL_PATH
    )
    
    # Process dataset
    report = pipeline.process_dataset()
    
    # Print summary
    print("\n" + "="*50)
    print("MILK10K PROCESSING COMPLETE")
    print("="*50)
    print(f"Total images processed: {report['dataset_info']['total_images_found']}")
    print(f"Successful segmentations: {report['processing_stats']['successful_segmentations']}")
    print(f"Successful classifications: {report['processing_stats']['successful_classifications']}")
    
    if report['accuracy_metrics']['total_evaluated'] > 0:
        print(f"Overall accuracy: {report['accuracy_metrics']['overall_accuracy']:.2%}")
    
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
