"""
Road Buddy VQA - Main Orchestration Script

End-to-end pipeline combining:
- Ingestion: Video frame sampling
- Perception: Query-guided keyframe selection  
- Reasoning: Gemini VLM for answer generation
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Setup project path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_config, get_vlm_config
from src.perception.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig
from src.reasoning.vlm_client import create_vlm_client
from src.reasoning.prompt_builder import build_mcq_prompt, format_context, PromptTemplate, PromptStyle
from src.reasoning.answer_extractor import extract_answer
from src.evaluation.dataset_loader import RoadBuddyDataset, load_dataset, VQASample
from src.evaluation.metrics import compute_accuracy, compute_accuracy_by_letter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from VQA pipeline."""
    sample_id: str
    question: str
    predicted_answer: str
    ground_truth: Optional[str] = None
    vlm_response: str = ""
    keyframe_count: int = 0
    processing_time_ms: float = 0.0
    
    @property
    def is_correct(self) -> bool:
        if not self.ground_truth:
            return False
        return self.predicted_answer[0] == self.ground_truth[0]


class RoadBuddyPipeline:
    """
    Complete VQA pipeline for Road Buddy Challenge.
    
    Workflow:
    1. Select keyframes from video using query-guided perception
    2. Build prompt with question and choices
    3. Generate answer using Gemini VLM
    4. Extract and return answer
    """
    
    def __init__(
        self,
        num_keyframes: int = 8,
        vlm_model: str = None,  # Loaded from config/models.yaml if None
        use_yolo: bool = False,
        prompt_style: str = "simple",
        verbose: bool = False,
        backend: str = None  # Loaded from config/models.yaml if None
    ):
        """
        Initialize the pipeline.
        
        Args:
            num_keyframes: Number of keyframes to select per video
            vlm_model: Qwen model name
            use_yolo: Whether to run YOLO detection
            prompt_style: Prompt template style
            verbose: Enable verbose logging
            backend: Inference backend ("transformers", "vllm", or "gemini" for API-based)
        """
        self.config = get_config()
        self.verbose = verbose
        
        # Load VLM defaults from centralized config if not specified
        vlm_config = get_vlm_config()
        if vlm_model is None:
            vlm_model = vlm_config.get("default", "qwen2.5-vl-7b-awq")
        if backend is None:
            backend = vlm_config.get("backend", "vllm")
        
        # Handle gemini backend - use Gemini model for API-based inference (no GPU needed)
        if backend == "gemini":
            vlm_model = "gemini-2.0-flash"
            logger.info("Using Gemini API backend - no local GPU required")
        
        # Initialize Keyframe Selector
        selector_config = KeyframeSelectorConfig(
            num_keyframes=num_keyframes,
            query_strategy="keyword",
            scoring_strategy="clip",
            yolo_mode="selected_only" if use_yolo else "none",
            use_translation=True,
            selection_strategy="diverse_top_k",
        )
        self.keyframe_selector = KeyframeSelector(selector_config)
        
        # Initialize VLM Client (Gemini)
        api_key = self.config.reasoning.api_key
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in environment. Please set it in .env file.")
        
        self.vlm_client = create_vlm_client(
            model_name=vlm_model,
            api_key=api_key,
            max_tokens=256,
            temperature=0.1,
            backend=backend
        )
        
        # Prompt template
        style_map = {
            "simple": PromptStyle.SIMPLE,
            "detailed": PromptStyle.DETAILED,
            "cot": PromptStyle.COT,
        }
        self.prompt_template = PromptTemplate(
            style=style_map.get(prompt_style, PromptStyle.SIMPLE),
            language="vi"
        )
        
        logger.info(f"Pipeline initialized: keyframes={num_keyframes}, model={vlm_model}")
    
    def process_single(self, video_path: str, question: str, choices: List[str], sample_id: str = "", ground_truth: Optional[str] = None) -> PipelineResult:
        """Process a single video question."""
        start_time = time.perf_counter()
        
        # Select keyframes
        if self.verbose:
            logger.info(f"Selecting keyframes from: {video_path}")
        
        selection_result = None
        try:
            selection_result = self.keyframe_selector.select(video_path, question)
            keyframes = selection_result.frames
            
            if self.verbose:
                logger.info(f"Selected {len(keyframes)} keyframes")
        except Exception as e:
            logger.error(f"Keyframe selection failed: {e}")
            keyframes = []
        
        # Build prompt
        detections = []
        if selection_result and selection_result.keyframes:
            for kf in selection_result.keyframes:
                if kf.detections:
                    for det in kf.detections.detections[:3]:
                        detections.append(f"{det.class_name}")
        
        context = format_context(
            detections=detections[:5] if detections else None,
            target_objects=selection_result.query_analysis.target_objects[:3] if selection_result else None
        )
        
        prompt = build_mcq_prompt(
            question=question,
            choices=choices,
            context=context,
            template=self.prompt_template
        )
        
        # Unload perception models to free memory for VLM
        if self.keyframe_selector:
            self.keyframe_selector.unload_models()
        
        # Generate answer with VLM
        if self.verbose:
            logger.info(f"Generating answer with {self.vlm_client.config.model}...")
        
        # Load model if needed (for single sample processing)
        if hasattr(self.vlm_client, 'load_model'):
            self.vlm_client.load_model()

        if keyframes and self.vlm_client.is_available():
            response = self.vlm_client.generate(keyframes, prompt)
            vlm_response = response.text
        else:
            vlm_response = "A"  # Default fallback
            logger.warning("VLM not available or no keyframes, using default answer")
        
        # Extract answer
        extraction = extract_answer(vlm_response, choices)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        if self.verbose:
            logger.info(f"Answer: {extraction.letter}")
        
        return PipelineResult(
            sample_id=sample_id,
            question=question,
            predicted_answer=extraction.letter,
            ground_truth=ground_truth,
            vlm_response=vlm_response,
            keyframe_count=len(keyframes) if keyframes else 0,
            processing_time_ms=processing_time
        )
    
    def process_sample(self, sample: VQASample) -> PipelineResult:
        """Process a VQASample from dataset."""
        return self.process_single(
            video_path=str(sample.video_abs_path),
            question=sample.question,
            choices=sample.choices,
            sample_id=sample.id,
            ground_truth=sample.answer
        )
    
    def evaluate(self, dataset: RoadBuddyDataset, max_samples: Optional[int] = None, save_results: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate pipeline on dataset using batch processing to save memory."""
        samples = dataset.samples[:max_samples] if max_samples else dataset.samples
        
        logger.info(f"Evaluating on {len(samples)} samples...")
        
        # --- Phase 1: Keyframe Selection (CLIP/YOLO) ---
        logger.info("=== Phase 1: Keyframe Selection ===")
        
        # Ensure VLM is unloaded
        if hasattr(self.vlm_client, 'unload_model'):
            self.vlm_client.unload_model()
            
        intermediate_results = []
        
        from tqdm import tqdm
        for sample in tqdm(samples, desc="Selecting Keyframes"):
            try:
                # process_single logic for keyframe selection
                video_path = str(sample.video_abs_path)
                question = sample.question
                
                if self.verbose:
                    logger.info(f"Selecting keyframes from: {video_path}")
                
                selection_result = None
                keyframes = []
                try:
                    selection_result = self.keyframe_selector.select(video_path, question)
                    keyframes = selection_result.frames
                    if self.verbose:
                        logger.info(f"Selected {len(keyframes)} keyframes")
                except Exception as e:
                    logger.error(f"Keyframe selection failed for {sample.id}: {e}")
                
                # Build context
                detections = []
                if selection_result and selection_result.keyframes:
                    for kf in selection_result.keyframes:
                        if kf.detections:
                            for det in kf.detections.detections[:3]:
                                detections.append(f"{det.class_name}")
                
                context = format_context(
                    detections=detections[:5] if detections else None,
                    target_objects=selection_result.query_analysis.target_objects[:3] if selection_result else None
                )
                
                prompt = build_mcq_prompt(
                    question=question,
                    choices=sample.choices,
                    context=context,
                    template=self.prompt_template
                )
                
                intermediate_results.append({
                    "sample": sample,
                    "keyframes": keyframes,
                    "prompt": prompt,
                    "keyframe_count": len(keyframes)
                })
                
            except Exception as e:
                logger.error(f"Error preparing sample {sample.id}: {e}")
        
        # Unload KeyframeSelector models!
        logger.info("Unloading perception models...")
        if self.keyframe_selector:
            self.keyframe_selector.unload_models()
        
        # Explicit GC between phases
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        time.sleep(2)  # Give system a moment to reclaim memory
        
        # --- Phase 2: VLM Generation (Qwen) ---
        logger.info("=== Phase 2: VLM Generation ===")
        
        # Load VLM
        if hasattr(self.vlm_client, 'load_model'):
            logger.info("Loading VLM model...")
            self.vlm_client.load_model()
            
        results = []
        predictions = []
        ground_truths = []
        
        for item in tqdm(intermediate_results, desc="Generating Answers"):
            sample = item["sample"]
            keyframes = item["keyframes"]
            prompt = item["prompt"]
            start_time = time.perf_counter()
            
            try:
                # Generate answer
                if keyframes and self.vlm_client.is_available():
                    response = self.vlm_client.generate(keyframes, prompt)
                    vlm_response = response.text
                else:
                    vlm_response = "A"
                    logger.warning(f"Using default answer for {sample.id}")
                
                # Extract answer
                extraction = extract_answer(vlm_response, sample.choices)
                
                processing_time = (time.perf_counter() - start_time) * 1000
                
                # Create result
                result = PipelineResult(
                    sample_id=sample.id,
                    question=sample.question,
                    predicted_answer=extraction.letter,
                    ground_truth=sample.answer_letter,
                    vlm_response=vlm_response,
                    keyframe_count=item["keyframe_count"],
                    processing_time_ms=processing_time
                )
                
                results.append(result)
                predictions.append(result.predicted_answer)
                ground_truths.append(sample.answer_letter)
                
                if self.verbose:
                    logger.info(f"Sample {sample.id}: {extraction.letter} (GT: {sample.answer_letter})")
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample.id}: {e}")
                predictions.append("A")
                ground_truths.append(sample.answer_letter)
        
        # Unload VLM when done
        if hasattr(self.vlm_client, 'unload_model'):
            self.vlm_client.unload_model()
            
        accuracy = compute_accuracy_by_letter(predictions, ground_truths)
        
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        
        metrics = {
            "total_samples": len(samples),
            "correct": correct,
            "accuracy": accuracy,
            "avg_processing_time_ms": sum(r.processing_time_ms for r in results) / len(results) if results else 0,
        }
        
        if save_results:
            output = {
                "metrics": metrics,
                "results": [
                    {
                        "id": r.sample_id,
                        "question": r.question,
                        "predicted": r.predicted_answer,
                        "ground_truth": r.ground_truth,
                        "correct": r.is_correct,
                        "time_ms": r.processing_time_ms
                    }
                    for r in results
                ]
            }
            with open(save_results, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {save_results}")
        
        return metrics
    
    def generate_submission(self, test_json_path: str, output_path: str) -> None:
        """Generate submission file for test set."""
        with open(test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        samples = test_data.get("data", test_data)
        test_dir = Path(test_json_path).parent
        
        logger.info(f"Generating submission for {len(samples)} test samples...")
        
        submissions = []
        
        from tqdm import tqdm
        for item in tqdm(samples, desc="Processing test"):
            video_path = test_dir / "videos" / item["video"]
            
            result = self.process_single(
                video_path=str(video_path),
                question=item["question"],
                choices=item["choices"],
                sample_id=item.get("id", "")
            )
            
            submissions.append({
                "id": item.get("id", ""),
                "answer": result.predicted_answer
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(submissions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Submission saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Road Buddy VQA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--evaluate', action='store_true', help='Run evaluation on train set')
    mode_group.add_argument('--submit', action='store_true', help='Generate submission for test set')
    mode_group.add_argument('--video', type=str, help='Process single video')
    
    # Single video options
    parser.add_argument('--question', type=str, help='Question for single video')
    parser.add_argument('--choices', nargs='+', help='Answer choices')
    
    # Evaluation options
    parser.add_argument('--train_json', type=str, default='data/raw/train/train.json', help='Path to train.json')
    parser.add_argument('--max_samples', type=int, help='Max samples for evaluation')
    parser.add_argument('--save_results', type=str, help='Path to save evaluation results')
    
    # Submission options
    parser.add_argument('--test_json', type=str, help='Path to test.json')
    parser.add_argument('--output', type=str, default='submission.json', help='Output submission file')
    
    # Pipeline options
    parser.add_argument('--keyframes', type=int, default=8, help='Number of keyframes')
    parser.add_argument('--model', type=str, default='qwen2.5-vl-7b', help='VLM model name')
    parser.add_argument('--use_yolo', action='store_true', help='Enable YOLO detection')
    parser.add_argument('--prompt_style', type=str, default='simple', choices=['simple', 'detailed', 'cot'])
    parser.add_argument('--backend', type=str, default='transformers', choices=['transformers', 'vllm', 'gemini'], help='Inference backend (gemini for API-based, no GPU needed)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RoadBuddyPipeline(
        num_keyframes=args.keyframes,
        vlm_model=args.model,
        use_yolo=args.use_yolo,
        prompt_style=args.prompt_style,
        verbose=args.verbose,
        backend=args.backend
    )
    
    if args.video:
        # Single video mode
        if not args.question or not args.choices:
            parser.error("--question and --choices required with --video")
        
        result = pipeline.process_single(
            video_path=args.video,
            question=args.question,
            choices=args.choices
        )
        
        print("Result")
        print("." * 50)
        print(f"Question: {result.question}")
        print(f"Answer: {result.predicted_answer}")
        print(f"Processing Time: {result.processing_time_ms:.1f} ms")
        print(f"Keyframes Used: {result.keyframe_count}")
        print(f"\nVLM Response:\n{result.vlm_response}")
        
    elif args.evaluate:
        dataset = load_dataset(args.train_json, validate_videos=True)
        
        metrics = pipeline.evaluate(
            dataset=dataset,
            max_samples=args.max_samples,
            save_results=args.save_results
        )
        
        print("Evaluation results")
        print("." * 50)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Correct: {metrics['correct']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Avg Processing Time: {metrics['avg_processing_time_ms']:.1f} ms")
        
    elif args.submit:
        if not args.test_json:
            parser.error("--test_json required with --submit")
        
        pipeline.generate_submission(
            test_json_path=args.test_json,
            output_path=args.output
        )

if __name__ == "__main__":
    main()