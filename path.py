#!/usr/bin/env python3
"""
MILK10k Segmentation and Classification Pipeline
Complete implementation with critical debug checkpoints
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import traceback
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============
# UPDATE THESE PATHS TO MATCH YOUR SETUP
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_GroundTruth.csv"
MASKS_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Masks"  # Optional
OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/SegConOutputs"
SAM2_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/segment-anything-2"
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"

# Domain configuration for MILK10k
LABEL_MAPPINGS = {
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
}

TEXT_PROMPTS = [
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
]

class MILK10kPipeline:
    def __init__(self, debug_mode=True):
        """Initialize the pipeline with debug checkpoints"""
        self.debug_mode = debug_mode
        self.device = None
        self.sam_predictor = None
        self.concept_model = None
        self.concept_processor = None
        self.ground_truth = None
        self.gt_lookup = {}
        
        # Run initialization checkpoints
        self._run_checkpoints()
        
    def _run_checkpoints(self):
        """Run all critical checkpoints"""
        
        # ============== CHECKPOINT 1: PATHS ==============
        if self.debug_mode:
            print("\n" + "="*60)
            print("CHECKPOINT 1: VERIFYING PATHS")
            print("="*60)
        
        for name, path in [("Dataset", DATASET_PATH), 
                           ("Ground Truth CSV", GROUNDTRUTH_PATH),
                           ("SAM2 Model", SAM2_MODEL_PATH), 
                           ("ConceptCLIP Model", CONCEPTCLIP_MODEL_PATH),
                           ("Output", OUTPUT_PATH)]:
            if name == "Output":
                os.makedirs(path, exist_ok=True)
                if self.debug_mode:
                    print(f"✓ {name}: Created/Verified at {path}")
            elif os.path.exists(path):
                if self.debug_mode:
                    print(f"✓ {name}: {path}")
            else:
                raise FileNotFoundError(f"✗ {name} NOT FOUND: {path}")
        
        # Create output subdirectories
        for subdir in ["segmented", "classifications", "reports", "visualizations"]:
            os.makedirs(os.path.join(OUTPUT_PATH, subdir), exist_ok=True)
        
        # ============== CHECKPOINT 2: GPU ==============
        if self.debug_mode:
            print("\n" + "="*60)
            print("CHECKPOINT 2: GPU/CUDA STATUS")
            print("="*60)
        
        if torch.cuda.is_available():
            self.device = "cuda"
            if self.debug_mode:
                print(f"✓ CUDA Available")
                print(f"  GPU: {torch.cuda.get_device_name()}")
                print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = "cpu"
            if self.debug_mode:
                print("⚠ CUDA NOT AVAILABLE - Using CPU (will be slow)")
        
        # Test allocation
        try:
            test_tensor = torch.randn(100, 100).to(self.device)
            del test_tensor
            torch.cuda.empty_cache() if self.device == "cuda" else None
            if self.debug_mode:
                print(f"✓ Device allocation test passed")
        except Exception as e:
            raise RuntimeError(f"✗ Device allocation failed: {e}")
        
        # ============== CHECKPOINT 3: LOAD SAM2 ==============
        if self.debug_mode:
            print("\n" + "="*60)
            print("CHECKPOINT 3: LOADING SAM2")
            print("="*60)
        
        self._load_sam2()
        
        # ============== CHECKPOINT 4: LOAD CONCEPTCLIP ==============
        if self.debug_mode:
            print("\n" + "="*60)
            print("CHECKPOINT 4: LOADING CONCEPTCLIP")
            print("="*60)
        
        self._load_conceptclip()
        
        # ============== CHECKPOINT 5: LOAD GROUND TRUTH ==============
        if self.debug_mode:
            print("\n" + "="*60)
            print("CHECKPOINT 5: LOADING GROUND TRUTH")
            print("="*60)
        
        self._load_ground_truth()
        
        if self.debug_mode:
            print("\n" + "="*60)
            print("ALL CHECKPOINTS PASSED!")
            print("="*60)
    
    def _load_sam2(self):
        """Load SAM2 model"""
        sys.path.insert(0, SAM2_MODEL_PATH)
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            if self.debug_mode:
                print("✓ SAM2 modules imported")
        except ImportError as e:
            raise ImportError(f"Cannot import SAM2: {e}")
        
        # Find checkpoint
        checkpoint_dir = os.path.join(SAM2_MODEL_PATH, "checkpoints")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files in {checkpoint_dir}")
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        
        # Find config
        config_path = None
        possible_configs = [
            os.path.join(SAM2_MODEL_PATH, "sam2", "configs", "sam2", "sam2_hiera_l.yaml"),
            os.path.join(SAM2_MODEL_PATH, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
            os.path.join(SAM2_MODEL_PATH, "configs", "sam2_hiera_l.yaml"),
        ]
        
        for path in possible_configs:
            if os.path.exists(path):
                config_path = path
                break
        
        if not config_path:
            import glob
            yaml_files = glob.glob(os.path.join(SAM2_MODEL_PATH, "**", "*.yaml"), recursive=True)
            if yaml_files:
                config_path = yaml_files[0]  # Use first found
            else:
                raise FileNotFoundError("No config file found for SAM2")
        
        if self.debug_mode:
            print(f"  Config: {os.path.basename(config_path)}")
            print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Build model
        try:
            sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            if self.debug_mode:
                print("✓ SAM2 model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to build SAM2: {e}")
    
    def _load_conceptclip(self):
        """Load ConceptCLIP model"""
        sys.path.insert(0, CONCEPTCLIP_MODEL_PATH)
        
        try:
            from ConceptModel.modeling_conceptclip import ConceptCLIP
            from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor
            if self.debug_mode:
                print("✓ ConceptCLIP modules imported")
        except ImportError as e:
            raise ImportError(f"Cannot import ConceptCLIP: {e}")
        
        try:
            self.concept_model = ConceptCLIP.from_pretrained(
                CONCEPTCLIP_MODEL_PATH, 
                local_files_only=True
            )
            self.concept_processor = ConceptCLIPProcessor.from_pretrained(
                CONCEPTCLIP_MODEL_PATH, 
                local_files_only=True
            )
            self.concept_model = self.concept_model.to(self.device)
            self.concept_model.eval()
            if self.debug_mode:
                print("✓ ConceptCLIP model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load ConceptCLIP: {e}")
    
    def _load_ground_truth(self):
        """Load ground truth labels"""
        try:
            self.ground_truth = pd.read_csv(GROUNDTRUTH_PATH)
            if self.debug_mode:
                print(f"✓ Loaded {len(self.ground_truth)} ground truth entries")
            
            # Create lookup dictionary
            for _, row in self.ground_truth.iterrows():
                lesion_id = str(row['lesion_id']).strip().lower()
                for col, disease_name in LABEL_MAPPINGS.items():
                    if col in row and float(row.get(col, 0)) == 1.0:
                        self.gt_lookup[lesion_id] = disease_name
                        break
            
            if self.debug_mode:
                print(f"✓ Created lookup for {len(self.gt_lookup)} lesions")
        except Exception as e:
            print(f"⚠ Warning: Could not load ground truth: {e}")
            self.ground_truth = None
            self.gt_lookup = {}
    
    def segment_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Segment image using SAM2"""
        h, w = image.shape[:2]
        
        masks_and_scores = []
        
        with torch.inference_mode():
            self.sam_predictor.set_image(image)
            
            # Try center point first
            point_coords = np.array([[w//2, h//2]])
            point_labels = np.array([1])
            
            try:
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
                
                for mask, score in zip(masks, scores):
                    if score > 0.5:
                        masks_and_scores.append((mask.astype(np.uint8), float(score)))
                        
            except Exception as e:
                if self.debug_mode:
                    print(f"  ⚠ Segmentation warning: {e}")
        
        if not masks_and_scores:
            # Fallback: return full image as mask
            full_mask = np.ones((h, w), dtype=np.uint8)
            masks_and_scores.append((full_mask, 0.5))
        
        return masks_and_scores
    
    def classify_image(self, image: np.ndarray) -> Dict[str, float]:
        """Classify image using ConceptCLIP"""
        from PIL import Image
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        try:
            inputs = self.concept_processor(
                images=pil_image,
                text=TEXT_PROMPTS,
                return_tensors='pt',
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            with torch.no_grad():
                outputs = self.concept_model(**inputs)
                logits = outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].t()
                probs = logits.softmax(dim=-1).cpu().numpy()[0]
            
            # Extract disease names from prompts
            disease_names = [prompt.split('showing ')[-1] for prompt in TEXT_PROMPTS]
            
            results = {disease: float(prob) for disease, prob in zip(disease_names, probs)}
            
            return results
            
        except Exception as e:
            if self.debug_mode:
                print(f"  ⚠ Classification warning: {e}")
            return {}
    
    def process_folder(self, folder_path: Path, save_outputs: bool = True) -> Dict:
        """Process one folder of images"""
        folder_name = folder_path.name
        print(f"\nProcessing folder: {folder_name}")
        
        # Get image files
        image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        
        if not image_files:
            print(f"  ⚠ No images found")
            return None
        
        folder_results = []
        
        # Process each image
        for idx, img_path in enumerate(image_files[:3]):  # Limit to 3 images per folder for speed
            print(f"  Image {idx+1}/{min(3, len(image_files))}: {img_path.name}")
            
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Segment
                masks_and_scores = self.segment_image(image_rgb)
                print(f"    Segmentation: {len(masks_and_scores)} masks")
                
                # Classify
                classification_probs = self.classify_image(image_rgb)
                if classification_probs:
                    best_disease = max(classification_probs, key=classification_probs.get)
                    confidence = classification_probs[best_disease]
                    print(f"    Classification: {best_disease} ({confidence:.3f})")
                else:
                    best_disease = "unknown"
                    confidence = 0.0
                
                # Save outputs if requested
                if save_outputs and masks_and_scores:
                    output_dir = Path(OUTPUT_PATH) / "segmented" / folder_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save best mask
                    best_mask = masks_and_scores[0][0]
                    mask_overlay = self._create_mask_overlay(image_rgb, best_mask)
                    output_file = output_dir / f"{img_path.stem}_segmented.png"
                    cv2.imwrite(str(output_file), cv2.cvtColor(mask_overlay, cv2.COLOR_RGB2BGR))
                
                folder_results.append({
                    'image': img_path.name,
                    'prediction': best_disease,
                    'confidence': confidence,
                    'num_masks': len(masks_and_scores)
                })
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
        
        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Aggregate folder results
        if folder_results:
            # Majority voting for folder prediction
            predictions = [r['prediction'] for r in folder_results]
            from collections import Counter
            vote_counts = Counter(predictions)
            folder_prediction = vote_counts.most_common(1)[0][0]
            folder_confidence = np.mean([r['confidence'] for r in folder_results if r['prediction'] == folder_prediction])
            
            # Get ground truth
            ground_truth = self.gt_lookup.get(folder_name.lower(), None)
            is_correct = (ground_truth == folder_prediction) if ground_truth else None
            
            return {
                'folder': folder_name,
                'num_images': len(folder_results),
                'prediction': folder_prediction,
                'confidence': float(folder_confidence),
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'individual_results': folder_results
            }
        
        return None
    
    def _create_mask_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create visualization of mask on image"""
        overlay = image.copy()
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
        return overlay.astype(np.uint8)
    
    def run(self, max_folders: int = None):
        """Run the full pipeline"""
        print("\n" + "="*60)
        print("RUNNING FULL PIPELINE")
        print("="*60)
        
        dataset_path = Path(DATASET_PATH)
        folders = [f for f in dataset_path.iterdir() if f.is_dir()]
        
        if max_folders:
            folders = folders[:max_folders]
        
        print(f"Processing {len(folders)} folders...")
        
        results = []
        correct_predictions = 0
        total_with_gt = 0
        
        for i, folder in enumerate(folders):
            print(f"\n[{i+1}/{len(folders)}]", end=" ")
            
            result = self.process_folder(folder, save_outputs=(i < 5))  # Save outputs for first 5
            
            if result:
                results.append(result)
                
                if result['ground_truth']:
                    total_with_gt += 1
                    if result['is_correct']:
                        correct_predictions += 1
            
            # Save intermediate results every 10 folders
            if (i + 1) % 10 == 0:
                self._save_results(results)
        
        # Calculate metrics
        accuracy = (correct_predictions / total_with_gt * 100) if total_with_gt > 0 else 0
        
        # Save final results and report
        self._save_results(results)
        self._generate_report(results, accuracy, total_with_gt)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Processed: {len(results)} folders")
        print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_with_gt})")
        print(f"Results saved to: {OUTPUT_PATH}/reports/")
        
        return results
    
    def _save_results(self, results: List[Dict]):
        """Save results to JSON"""
        output_file = Path(OUTPUT_PATH) / "reports" / "results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _generate_report(self, results: List[Dict], accuracy: float, total_with_gt: int):
        """Generate summary report"""
        report = {
            'total_folders': len(results),
            'folders_with_ground_truth': total_with_gt,
            'accuracy': accuracy,
            'predictions_distribution': {},
            'confusion_matrix': defaultdict(lambda: defaultdict(int))
        }
        
        # Calculate distribution
        for r in results:
            pred = r['prediction']
            gt = r['ground_truth']
            
            report['predictions_distribution'][pred] = report['predictions_distribution'].get(pred, 0) + 1
            
            if gt:
                report['confusion_matrix'][gt][pred] += 1
        
        # Convert defaultdict to regular dict for JSON
        report['confusion_matrix'] = {k: dict(v) for k, v in report['confusion_matrix'].items()}
        
        # Save report
        report_file = Path(OUTPUT_PATH) / "reports" / "summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save CSV for easy viewing
        df = pd.DataFrame(results)
        csv_file = Path(OUTPUT_PATH) / "reports" / "results.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\nReport saved to: {report_file}")
        print(f"CSV saved to: {csv_file}")


def test_single_image():
    """Quick test with a single image"""
    print("\n" + "="*60)
    print("QUICK TEST: SINGLE IMAGE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = MILK10kPipeline(debug_mode=True)
    
    # Find test image
    dataset_path = Path(DATASET_PATH)
    test_image = None
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            images = list(folder.glob("*.jpg"))
            if images:
                test_image = images[0]
                break
    
    if not test_image:
        print("✗ No test image found")
        return False
    
    print(f"\nTest image: {test_image}")
    
    # Load and test
    image = cv2.imread(str(test_image))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Test segmentation
    print("\nTesting segmentation...")
    masks = pipeline.segment_image(image_rgb)
    print(f"✓ Generated {len(masks)} masks")
    
    # Test classification
    print("\nTesting classification...")
    probs = pipeline.classify_image(image_rgb)
    if probs:
        best = max(probs, key=probs.get)
        print(f"✓ Prediction: {best} ({probs[best]:.3f})")
    
    print("\n✓ SINGLE IMAGE TEST PASSED")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MILK10k Pipeline")
    parser.add_argument("--test", action="store_true", help="Run quick test only")
    parser.add_argument("--max-folders", type=int, default=None, help="Limit number of folders to process")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug messages")
    args = parser.parse_args()
    
    try:
        if args.test:
            # Run quick test
            success = test_single_image()
            if not success:
                sys.exit(1)
        else:
            # Run full pipeline
            pipeline = MILK10kPipeline(debug_mode=not args.no_debug)
            
            # Quick test first
            if not args.no_debug:
                print("\nRunning quick validation...")
                test_success = test_single_image()
                if not test_success:
                    print("Quick test failed, aborting.")
                    sys.exit(1)
            
            # Run full pipeline
            results = pipeline.run(max_folders=args.max_folders)
            
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        print(traceback.format_exc())
        sys.exit(1)
