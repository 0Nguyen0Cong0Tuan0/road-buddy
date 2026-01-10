# **Strategic Blueprint for Zalo AI Challenge 2025: The RoadBuddy Autonomous Assistant**

# **1. Executive Summary**
The Zalo AI Challenge 2025, specifically the RoadBuddy track, represents a watershed moment in the intersection of computer vision, legal reasoning, and edge-constrained artificial intelligence within the Vietnamese technological landscape.

The mandate is clear yet technically formidable: engineer a driving assistant capable of ingesting high-dimensional dashcam video data (5 to 15 seconds in duration), processing it through a neuro-symbolic reasoning pipeline, and delivering legally grounded answers to traffic inquiries strictly adhering to the Vietnamese legal framework. This must be achieved within a stringent inference window of 30 seconds on consumer-grade hardware—specifically a single NVIDIA RTX 3090 or A30 GPU—while adhering to a model parameter ceiling of 9 billion.

This report articulates a comprehensive, research-backed strategy to secure a winning position in this competition. Our analysis suggests that a monolithic approach—relying on a single, end-to-end Vision-Language Model (VLM)—is fundamentally ill-suited to the dual pressures of high-fidelity perception (detecting small traffic signs) and precise legal citation (avoiding hallucination of Decree 100/2019 or Law 36/2024). Standard VLMs, when constrained to 9B parameters, typically lack the visual resolution to discern distant signage in wide-angle footage and the long-context retention required to memorize the entirety of the Vietnam Road Traffic Order and Safety Law.

Consequently, we propose the **'Lenses & Law'** Architecture: a hybrid, modular-augmented pipeline that decouples perception from reasoning. This system synthesizes a lightweight, high-recall detection front-end (YOLOv11-Nano) with a retrieval-augmented generation (RAG) core utilizing **Qwen2.5-VL-7B-Instruct** or **MiniCPM-V 2.6** as the executive reasoning engine. By treating traffic sign recognition as a retrieval task rather than a generation task, we dramatically reduce the cognitive load on the VLM, allowing it to focus on complex causal reasoning and legal interpretation. Futhermore, to satisfy the 30-second latency constraint, we introduce a dynamic keyframe selection mechanism driven by optical flow analysis, rejecting redundant temporal data to maximize the information density of the tokens fed into the VLM.

The following sections detail the theoretical underpinnings, architectural decisions, and engineering optimizations required to realize this vision. We draw extensively from recent benchmarks in multimodal learning, advancements in quantization (AWQ), and the specific legal texts governing Vietnamese road traffic to construct a robust, hallucination-resistant, and high-performance system.
# **2. Operational Context and Constraint Decomposition**
Success in the RoadBuddy challenge requires a rigorous deconstruction of the operational environment. The constraints imposed—time, memory, and model size—are not merely hurdles but defining features that dictate the architectural topology. We must analyze the hardware capabilities of the RTX 3090/A30 against the computational demands of processing video data and generating text.
## **2.1. Hardware Envelope: The RTX 3090 and A30**
The target deployment environment is specified as a single NVIDIA RTX 3090 or A30 GPU. Understanding the microarchitecture of these cards is essential for optimizing the inference pipeline. The RTX 3090 is built on the Ampere architecture, featuring 24 GB of GDDR6X memory with a massive bandwidth of 936 GB/s. This high bandwidth is a critical asset for memory-bound applications like Large Language Model (LLM) inference, where the bottleneck is often moving weights from memory to compute units rather than the computation itself.

However, the 24 GB VRAM limit poses a significant challenge for a 9B parameter model processing video. A standard 7B model in half-precision (FP16) consumes approximately 14-15 GB of VRAM for weights alone. This leaves less than 10 GB for the KV cache (which grows linearly with context length), the visual encoder's activation maps (which are heavy for video), and any auxiliary models like the object detector or vector database. Without quantization, the system would likely encounter Out-Of-Memory (OOM) errors when processing the 128K context window capabilities of modern VLMs or handling multi-frame video inputs.

The A30, alternatively, offers similar memory capacity (24 GB HBM2) but with significantly higher memory bandwidth (933 GB/s) and distinct Tensor Core optimizations for inference. While the A30 is a data center card optimized for throughput, the RTX 3090 is a consumer card with raw power. Our strategy must be agnostic to the subtle differences but optimized for the shared 24 GB VRAM constraint.
## **2.2. The 30-Second Inference Budget**
The 30-second latency limit is the most stringent constraint. In the context of video QA, latency is composed of four distinct phases:
1. **Video decoding & preprocessing** $\rightarrow$ Converting compressed MP4/H.264 data into raw tensor frames. On a CPU, decoding a 15-second 1080p video can take 1–3 seconds.
2. **Visual encoding** $\rightarrow$ Passing frames through the Vision Transformer (ViT). For a model like Qwen2-VL, which uses a deep ViT (e.g., SigLIP or similar), encoding a single image can take 50–200ms. Encoding 16 frames sequentially could consume 3–4 seconds.
3. **Retrieval (RAG)** $\rightarrow$ Querying the vector database. With an optimized index (e.g., HNSW), this is negligible (<100ms).
4. **Autoregressive generation** $\rightarrow$ The LLM generates the answer token by token. A 7B model on an RTX 3090 can generate 40–60 tokens per second. A detailed legal answer might require 300 tokens, consuming 5–8 seconds.

Summing these up ($3s + 4s + 0.1s + 8s \approx 15.1s$), we theoretically fit within the 30s limit. However, this assumes an optimized pipeline. Naive implementations using Hugging Face transformers defaults often result in 2x-3x slower performance due to lack of continuous batching and unoptimized attention kernels. Therefore, the use of a high-performance serving engine like vLLM is non-negotiable.
## **2.3. The 9B Parameter Ceiling**
The limitation to 9 billion parameters places the solution in the realm of Small Language Models (SLMs). While 70B+ models like GPT-4 or Qwen-72B exhibit emergent reasoning and vast world knowledge, 7B models often struggle with complex, multi-step reasoning and 'long-tail' knowledge recall.

In the context of Vietnamese traffic law, a 7B model pre-trained primarily on English or Chinese web data will likely hallucinate when asked about specific articles of Law 36/2024. It may conflate Vietnamese rules with US traffic laws (e.g., 'Right turn on red,' which is generally prohibited in Vietnam unless signaled). In the context of Vietnamese traffic law, a 7B model pre-trained primarily on English or Chinese web data will likely hallucinate when asked about specific articles of Law 36/2024. It may conflate Vietnamese rules with US traffic laws (e.g., 'Right turn on red,' which is generally prohibited in Vietnam unless signaled). 
# **3. Legal Knowledge Engineering and RAG Strategy**
The core requirement of the RoadBuddy challenge is to answer questions using Vietnamese law. This is not a general driving advice task; it is a legal compliance task. The system must cite specific regulatory documents. We must rigorously analyze the target legal corpus to structure our Retrieval-Augmented Generation (RAG) system effectively.
## **3.1. Analysis of the Legal Corpus**
The legal framework for Vietnamese road traffic has recently undergone a significant overhaul. The challenge explicitly references two key documents:

1 _ **Law No. 36/2024/QH15** on Road Traffic Order and Safety $\rightarrow$ Passed by the National Assembly on June 27, 2024, and effective from January 1, 2025. This law separates traffic rules from infrastructure management (which remains in the Road Law). 
- *Key changes* $\rightarrow$ It enforces a zero-tolerance alcohol ban (Article 9, Clause 2), strictly prohibiting driving with any alcohol concentration in blood or breath. This is a critical distinction from older laws or international standards that might allow small amounts. A model trained on data prior to mid-2024 might hallucinate the old 50mg/100ml limit unless corrected by RAG.   

- *Points system* $\rightarrow$ It introduces a license point deduction system (12 points total). The system must be aware of which violations trigger point deductions versus simple fines.

- *Child safety* $\rightarrow$ New rules for children under 10 years old and under 1.35m height, prohibiting them from sitting in the front seat.

2 _ **QCVN 41:2024/BGTVT** (National Technical Regulation on Road Signs) $\rightarrow$ This technical regulation defines the visual appearance and legal meaning of every traffic sign and road marking in Vietnam.
- *Complexity* $\rightarrow$ It contains hundreds of signs (Prohibitory P.x, Command R.x, Warning W.x). A generic VLM will identify a 'red circle' but may not distinguish between 'P.103a' (No cars) and 'P.103b' (No right turn for cars). The distinction is purely visual but legally vast.

## **3.2. Structuring the Legal Vector Database**
To enable the VLM to use this knowledge, we cannot simply 'dump' the PDF text into the prompt context limit, doing so would dilute attention and exceed token limits. We must structure a specialized **Legal Vector Database**.
### **3.2.1. Hybrid Indexing Strategy**
We propose a dual-index approach using a vector database like Qdrant or ChromaDB, which supports multimodal payloads.

**Index A: The Statutory Text Index (Law 36/2024)**
- **Chunking strategy** $\rightarrow$ We will chunk the law by **Article**. Each chunk will contain the full text of the Article, its Title, and the Chapter it belongs to.
- **Metadata** $\rightarrow$ `{'article_id': '9', 'chapter': 'General Rules', 'keywords': ['alcohol', 'drunk', 'ban']}`.
- **Embedding model** $\rightarrow$ We require a multilingual embedding model that excels in Vietnamese legal semantic retrievel. **bge-m3** (BAAI/bge-m3) is the current SOTA for multingual retrieval, supporting long contexts (8192 tokens) and dense/sparse retrieval.
- **Sparse retrieval** $\rightarrow$ We will also maintain a BM25 (keyword) index. If a user asks about 'Article 9', dense vector search might drift to semantically similar articles, whereas BM25 will lock onto the keyword 'Article 9' precisely.

**Index B: The Visual Sign Registry (QCVN 41)**
- **Content** $\rightarrow$ This index contains the standard reference images of all traffic signs defined in QCVN 41:2024/BGTVT, along with their legal descriptions.
- **Embedding model** $\rightarrow$ We use **CLIP (ViT-L/14)** or **SigLIP** to generate embeddings for every reference sign image.
- **Workflow** $\rightarrow$ When the system detects a sign in the dashcam video, it crops the sign, computes its visual embedding, and queries this Visual Sign Registry. The result is the exact sign code (e.g., 'P.102') and its legal meaning, which is then fed as text to the VLM. This bridges the gap between pixel data and legal text.

## **3.3. Prompt Engineering for Legal Compliance**

The retrieved legal text must be injected into the VLM using a strict system prompt to enforce compliance.

**System Prompt Template**:
```markdown
You are a legal expert in Vietnamese Road Traffic Law. You are provided with a video analysis and a set of retrieved legal articles from Law 36/2024/QH15 and QCVN 41:2024. Your task is to answer the user's question. You must cite the specific Article and Clause that supports your answer. If the video shows a violation, identify it strictly according to the retrieved laws. Do not use external knowledge if it conflicts with the provided context.
```

This formulation leverages 'In-Context Learning' to align the model's output with the specific constraints of the 2025 challenge, mitigating the risk of hallucination outdated 2008 laws.

# **4. Visual Perception: The 'Lenses'**
A driving assistant is only as good as its eyes. The Zalo challenge involves wide-angle dashcam footage where critical details—traffic signs, lane markings, and traffic light states—are often small and fleeting. Relying solely on the VLM's native encoder to process the entire video frames is inefficient and perceptually inadequate.

## **4.1. The 'Small Object' Dilemma** 
Standard VLMs typically resize input images to a fixed resolution, often $336 \times 336$ or $448 \times 448$ pixels. In a $1920 \times 1080$ dashcam video, a traffic sign might be $50 \times 50$ pixels. When downsampled to a $448 \times 448$ grid, this sign becomes a $10 \times 10$ feature map blob, effectively destroying the text or symbol required for identification. To read a 'Max Speed 60' sign, the model needs high-resolution access to that specific region.

## **4.2. The Modular Perception Pipeline**

We propose a **Coarse-to-Fine Perception Module** that acts as a pre-processor before the VLM.

### **4.2.1. Object Detection with YOLOv11**

We will deploy a lightweight object detection model, **YOLOv11-Nano** or **YOLOv10-Small**, trained specifically on Vietnamese traffic datasets (VNTS).
- **Why YOLO11?** $\rightarrow$ It offers the best trade-off between latency and accuracy. The Nano version runs in <10ms on an RTX 3090.
- **Target classes** $\rightarrow$ We do not need to classify every sign perfectly at this stage. We only need to detect them. We define a simplified class ontology: `Prohibitory`, `Warning`, `Command`, `Information`, `Traffic Light`, `Road Marking`.
- **Objective** $\rightarrow$ High recall. It is acceptable to propose a fase positive (which the VLM will later dicard) but fatal to miss a true positive.

### **4.2.2. The Crop-and-Zoom Mechanism**
Upon detecting a traffic sign with the YOLO model, the system performs a 'virtual zoom':
1. **Coordiate extraction** $\rightarrow$ Get the bounding box ($x,y,w,h$) from the high-resolution sourcec frame (1080p).
2. **Context expansion** $\rightarrow$ Expand the box by 20% to capture context (e.g., the pole or a supplementary sign below it).
3. **Crop & stack** $\rightarrow$ Crop these regions, If multiple signs are present, stitch them into a single 'Sign Summary Image'.
4. **VLM input** $\rightarrow$ The VLM receives two visual inputs:
    - **Context stream** $\rightarrow$ A sequence of downsampled full frames (for understanding vehicle motion, lane position).
    - **Detail stream** $\rightarrow$ The high-resolution 'Sign Summary Image' (for reading specific text and symbols).

This strategy allows a 7B model to achieve OCR performance comparable to a 72B model, as it is 'looking' at the critical pixels with full fidelity.

## **4.3. Dynamic Keyframe Selection**
Processing video at 30 fps is redudant. We need to select the most informative frames to stay within the 30s budget. Uniform sampling (e.g., every 30th frame) risks missing a ign that is visisble for only 0.5 seconds.

**Algorithm** $\rightarrow$ We implement a **Semantic Triggered Sampling** method.
- *Base rate* $\rightarrow$ Sample 1 frame per second (FPS) for general context.
- *Trigger* $\rightarrow$ If the YOLO detector's confidence for a `traffic_sign` class peaks or if the bounding box area exceeds a threshold (indicating the car is close to the sign), we forcefully sample that frame.
- *Deduplication* $\rightarrow$ Use optical flow or histogram comparison to reject frames that are $> 95\%$ similar to previously keyframe (e.g., stopped at a red light).

# **5. Vision-Language Model Architecture**
The reasoning engine - the 'Brain' - must synthesize the visual cues from the perception module and the legal facts from the RAG module to answer the user's question.

## **5.1. Comparative Model Selection**
We evaluated three primary candidates available in the <9B parameter class as of late 2024/early 2025:

### **5.1.1. Qwen2.5-VL-7B-Instruct**

**Architecture** $\rightarrow$ Qwen2.5-7B LLM + Naive Dynamic Resolution Vision Encoder (similar to SigLIP).

**Strengths**
- **Naive Dynamic Resolution** $\rightarrow$ It handles images of varying aspect ratios natively, mapping them to dynamic token counts. This is ideal of our 'Sign Summary Image' which might be a tall strip of stacked signs.

- **Strong OCR** $\rightarrow$ Benchmarks on DocVQA and MathVista show it outperforms competitors in reading text. This is vital for reading auxiliary sign plates (e.g., '6h-22h').

- **Long video support** $\rightarrow$ Capable of processing 20min+ videos, implying robust positional embeddings for our 15s clips.

**Weaknesses** $\rightarrow$ Video inference throughput in vLLM can be lower than image input due to complex attention patterns.

### **5.1.2. MiniCPM-V 2.6 (8B)**

**Architecture** $\rightarrow$ SigLIP-400M + Qwen2-7B.

**Strengths**
- **Token density** $\rightarrow$ It encodes high-resolution images (1.8M pixels) into only 640 tokens. This is significantly more efficient than standard ViT encodings, offering a speed advantage for the 30s limit.
- **Edge optimization** $\rightarrow$ Designed for end-side devices (iPads, smartphones), suggesting low VRAM footprint.

**Weaknesses** $\rightarrow$ While fast, its reasoning depth on complex causal traffic scenarios (e.g., 'Who has the right of way in this roundabout?') might lag behind the specialized Qwen2.5-VL.

### **5.1.3. InternVL2-8B**

**Architecture** $\rightarrow$ InternViT-300M + InternLM2.5-7B.

**Strengths** $\rightarrow$ SOTA on generic video benchmarks (Video-MME).

**Weaknesses** $\rightarrow$ The visual encoder is heavy (300M parameters vs. smaller alternatives), potentially impacting latency.

## **5.2. Selection: Qwen2.5-VL-7B-Instruct**
We select **Qwen2.5-VL-7B-Instruct**  the primary candidate. THe decisive factor is its **OCR capability** and **Dynamic Resolution**. In traffic sign analysis, reading the text on a sign is often the difference between a correct and incorrect legal interpretation. The potential latency disadvantage can be mitigated via quantization (AWQ) and the efficient keyframe selection strategy outlined in Section 4.3.
# **6. Data Strategy and Curation**
Data is the fuel for the AI engine. We must construct a robust training pipeline that bridges the gap between generic visual pre-training and specific Vietnamese legal reasoning.

## **6.1. Data Aggregation**
We will leverage a hierarchy of datasets to train the Perception Module (YOLO) and fine-tune the VLM.

### **6.1.1. The VLSP 2025 MLQA-TSR Dataset**
This is the gold standard for the challenge. It contains multimodal legal question-answering pairs specifically for Vietnamese traffic signs/scenarios.

- **Composition** $\rightarrow$  Images of traffic signs + Textutal Questions + Legal Articles + Correct Answers.
- **Utility** $\rightarrow$ This will be the primary validation set and the core of the instruction-tuning dataset for the VLM.

### **6.1.2. VNTS (Vietnamese Traffic Sign Dataset)**
A comprehensive dataset for object detection.

- **Size** $\rightarrow$ ~3200 images.
- **Classes** $\rightarrow$ Detailed breakdown of classes (e.g., `Speed limit (50km/h)`, `No parking`, `Pedestrian crossing`).
- **Utility** $\rightarrow$ Training the YOLOv11 detector. We will map the specific classes (e.g., `Speed limit (50km/h`) to broader super-classes (`Prohibitory`) for the detector, leaving the fine-grained reading to the VLM.

### **6.1.3. Zalo Traffic Sign Detection 2020**
A legacy dataset from Zalo's previous internal competitions.
- **Size** $\rightarrow$ 4500 images.
- **Features** $\rightarrow$ Contains challenging conditions (night, rain, occlusion).
- **Utility** $\rightarrow$ Augment the YOLO training set to improve robustness against environmental noise.

## **6.2. Synthetic Data Generation (The Data Flywheel)**
To teach the VLM 'reasoning' rather than just 'recognition', we need data that explains the *why*. We will generate this synthetically.

_ **Source material** $\rightarrow$ Unlabeled dashcam videos from Vietnamese sources (YouTube channels, open datasets).

_ **The 'Teacher' model** $\rightarrow$ We use a powerful proprietary model (e.g., GPT-4o or Qwen-Max) to annotate these videos.
- *Prompt* $\rightarrow$ 'Analyze this video frame-by-frame. Identify every traffic sign. Explain its meaning under Vietnamese Law 36/2024. Formulates a difficult multiple-choice question about the driver's obligation in this scene and provide the correct answer with legal citations.'

_ **Distillation** $\rightarrow$ This process generates thousands of high-quality (Video, Question, Answer, Chain-of-Thought) tuples. We use this synthetic dataset to fine-tune the Qwen2.5-VL-7B-Instruct model, effectively distilling the reasoning capabilities of the larger model into our constrained 7B architecture.

# **7. System Engineering and Optimization**

Meeting the 30-second inference limit on a single GPU requires optimization at the kernel and serving level. We cannot rely on standard Python inference loops.

## **7.1. Inference Engine: vLLM**
We will utilize **vLLM** (Virtual Large Language Model) as the serving backend. vLLM is the industry standard for high-throughput inference due to its **PagedAttention** algorithm, which optimizes memory management for the KV cache during autoregressive generation.

_ **Video handling in vLLM** $\rightarrow$ Recent updates to vLLM have added support for multi-modal inputs, including Qwen2-VL. However, processing video tokens can be slow if not configured correctly.

_ **Configuration** 
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --quantization awq \
    --dtyle half \
    --limit-mm-per-promtpt video=1, image=4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```
- *Limit MM* $\rightarrow$ We explicitly limit the number of video chunks and image crops to prevent OOM.
- *Max Model Len* $\rightarrow$ Reducing context from 32k to 8k saves significant VRAM for the KV cache, sufficient for 15s videos.

## **7.2. Activation-Aware Weight Quantization (AWQ)**
Running the model in FP16 consumes ~15GB VRAM. We will use **INT4 AWQ**.

_ **Mechanism** $\rightarrow$  AWQ identifies the 1% of salient weights that are critical for model performance and keeps them in higher precision, while quantizing the rest to 4-bit.

_ **Benefit** $\rightarrow$ This reduces the model size to ~5.5GB, freeing up 10GB for KV cache and visual encodings, while incurring only a ~1-2% accuracy drop.

_ **Performance** $\rightarrow$ On the RTX 3090, INT4 inference is significantly faster than FP16 due to higher effective memory bandwidth usage (moving 4 bits weights is 4x faster than 16 bits). This serves to double the token generation speed, crucial for the autoregressive generation phase.

## **7.3. Docker Containerization**
The solution musst be deployed as a Docker container.

_ **Base image** $\rightarrow$ `vllm/vllm-openai:latest` or a custom build with specific CUDA 12.1 optimizations.

_ **Dependency management** $\rightarrow$ All models (YOLO, Qwen-AWQ, Embedding models) must be baked into the Docker image to ensure offline functionality. We will use a multi-stage build to keep the iamge size manageable, although the model weights alone will make it large (~10-15GB).

_ **Runtime command** 
```bash
docker run --gpus all --shm-size=12g -v /path/to/models:/models roadbuddy-submission
```
(Note `--shm-size` is critical for NCCL/PyTorch communications).

# **8. Experimental Validation Plan**
To ensure the strategy is robust, we define a rigorous validation protocol using the metrics specified in the VLSP/Zalo challenge.

## **8.1. Metrics**
1. **Retrieval F2-score** $\rightarrow$ For the RAG component. We prioritize Recall (F2) over Precision, because falling to retrieve the relevant legal law is a catastrophic failure, whereas retrieving an extra irrelevant law is manageable.

2. **Accuracy** $\rightarrow$ For the final QA generation.

3. **Inference latency** $\rightarrow$ Measured as the wall-clock time from receiving the video path to returning the JSON response. Must be < 30.0 seconds.

## **8.2. Ablation Studies**
We will perform the following experiments to tune the system:
- **Keyframe count vs. accuracy** $\rightarrow$ Test 4,8,16 keyframes. Hypothesis: 8 frames is the diminishing returns point for traffic QA.
- **Crop vs. no-crop** $\rightarrow$ Evaluate the impact of the Crop-and-Zoom mechanism on OCR accuracy. We expect a >15% accuracy gain on small sign recognition.
- **Quantization impact** $\rightarrow$ Compare FP16 vs. AWQ INT4. Hypothesis: Accuracy drop is <2% but latency improves by 40%.

# **9. Conclusion**
The Zalo AI Challenge 2025 RoadBuddy track is a test of architectural ingenuity under constraint. Our proposed solution moves beyond the simplistic application of a pretrained model. Instead, it engineers a symbiotic system:
1. **Perception** $\rightarrow$ A dedicated YOLOv11 'eye' that foveates on critical traffic signs, overcoming the resolution limitations of standard VLMs.
2. **Knowledge** $\rightarrow$ A structured Legal RAG system that bridges the gap between pixel data and the rigid statutes of Vietnamese traffic law.
3. **Reasoning** $\rightarrow$ An optimized Qwen2.5-VL-7B engine, accelerated via AWQ and vLLM, to synthesize these inputs into legally compliant answers.

By rigoously optimizing for the 9B parameter and 30-second limitation through quantization, dynamic keyframe selection, and modular design, this strategy provides a definitive roadmap to achieving SOTA performance in the RoadBuddy challenge. The integration of specific Vietnamese datasets (VLSP, VNTS) and the generation of synthetic reasoning traces further ensures that the system is not just a generic AI, but a specialized, compliant Vietnamese driving assistant.

# **10. Data Tables and Implementation References**

**Table 1: Model Candidate Comparison (<9 Parameters)**

| Model Name | Parameter Count | Vision Encoder | Video Capability | OCR Performance | Recommended Use |
| --- | --- | --- | --- | --- | --- |
| **Qwen2.5-VL-7B** | 7.6B | Naive Dynamic Res | High (Long context) | SOTA (DocVQA) | Primary Reasoner |
| **MiniCPM-V 2.6** | 8B | SigLIP-400M | Moderate | High (Token Efficient) | Fallback / Speed |
| **InternVL2-8B** | 8B | InternViT-300M | High (Video-MME) | Moderate | Research |
| **LLaVA-NeXT-Video** | 7B | CLIP-ViT-L | High | Moderate | Baseline |

**Table 2: Dataset Utilization Strategy**

| Dataset Name | Source | Size | Usage Strategy| 
| --- | --- | --- | --- |
| **VLSP 2025 MLQA-TSR** | VLSP Challenge | ~3,000 QA Pairs | Validation / Fine-tuning (Gold Standard) | 
| **VNTS** | Kaggle | 3,200 Images | YOLO Training (Detection) | 
| **Zalo 2020** | Zalo AI | 4,500 Images | Augmentation (Robustness) | 
| **Law 36/2024 Corpus** | Official Text | 89 Articles | RAG Knowledge Base (Text Index) | 
| **QCVN 41 Sign Registry** | Official Reg | ~400 Sign Images | RAG Knowledge Base (Visual Index) |


**Table 3: Estimated Latency Budget (RTX 3090, Qwen2.5-VL-AWQ)**

| Pipeline Stage | Operation | Estimated Time | Optimization |
| --- | --- | --- | --- |
| **1. Preprocessing** | Video Decode + Optical Flow | 2.5s | OpenCV CUDA |
| **2. Detection** | YOLOv11 Inference (16 frames) | 0.5s | TensorRT / Batched |
| **3. Encoding** | Visual Encoder (ViT) | 4.0s | vLLM / Pipeline Parallel |
| **4. Retrieval** | Vector Search (Qdrant) | 0.2s | HNSW Index |
| **5. Generation** | LLM Decoding (256 tokens) | 6.5s | AWQ INT4 / vLLM |
| **Overhead** | Python/Docker Overhead | 1.0s | Efficient Code |
| Total |  | ~14.7s | Well within 30s Limit |

---

This research-backed repository architecture is designed to implement the **Lenses & Law Strategy** for the Zalo AI Challenge 2025. It is modeled after industry-standard repositories like **NVIDIA NeMo/Skills** and **Video-LLaVA**, but adapted for the strict constraints of the competition (Single GPU, <30s latency, <9B parameters).

The repository is named `RoadBuddy-Core`.

### **1. High-Level Architecture Overview**
The architecture follows a **Modular-Monolith** design pattern. Unlike end-to-end black boxes, this approch separates **Perception** (seeing), **Knowledge** (recalling law), and **Reasoning** (answering). This allows for independent debugging - critical when the system fails (e.g., 'Did it miss the sign, or did it misinterpret the law?').

**Data flow**
1 _ **Input** $\rightarrow$ Dashcam video ($V$) + User question ($Q$)

2 _ `perception_engine`
- **Detection** $\rightarrow$ Scans video using **YOLOv11-Nano** to find traffic signs.
- **Triage** $\rightarrow$ Uses **optical flow** (OpenCV) to reject static/redundant frames.
- **Output** $\rightarrow$ A sequence of 4-8 'Keyframes' ($K$) + cropped 'Sign Patches' ($S_p$).

3 _ `knowledge_retriever` **(RAG)**
- **Visual query** $\rightarrow$ Embeds $S_p$ (via CLIP/SigLIP) to query the **QCVN 41 Sign Registry**.
- **Text query** $\rightarrow$ Embeds $Q$ (via bge-m3) to query the **Law 36/2024 Corpus**.
- **Output** $\rightarrow$ JSON object containing relevant Legal Articles and Sign Definitions.

4 _ `reasoning_engine` **(VLM)**
- **Prompting** $\rightarrow$ Dynamic prompt builder injects $K$, $Q$, and Legal Context.
- **Inference** $\rightarrow$ **Qwen2.5-VL-7B-Instruct** (served via vLLM with AWQ quantization) generates the final answer.

5 _ **Output** $\rightarrow$ Final text answer strictly compliant with Vietnamese law.

### **2. Repository Directory Structure**
This structure uses Hydra for configuration management, allowing you to swap components (e.g., switch YOLOv11 for YOLOv10) without changing code.
```bash
RoadBuddy-Core/ 
├── configs/ # Hydra Configuration Files 
│ ├── config.yaml # Main entry point 
│ ├── perception/ 
│ │ ├── yolov11_nano.yaml # Fast detection config 
│ │ └── yolov10_small.yaml # Alternative detection config 
│ ├── vlm/ 
│ │ ├── qwen2.5_vl_awq.yaml # Primary reasoning engine (vLLM) 
│ │ └── minicpm_v2.6.yaml # Fallback edge model 
│ └── rag/ 
│ ├── qdrant_local.yaml # Vector DB config 
│ └── law_36_2024.yaml # Legal corpus settings 
├── data/ # Data Storage (Mounted via Docker) 
│ ├── raw/ # Videos from Zalo/VLSP 
│ ├── processed/ # Extracted frames/patches 
│ ├── knowledge_base/ # PDF/JSON of Laws (Law 36, QCVN 41) 
│ └── vector_store/ # Persisted Qdrant/Chroma indices 
├── docker/ # Deployment 
│ ├── Dockerfile # Submission container 
│ ├── entrypoint.sh # Runtime script 
│ └── requirements.txt # Python dependencies 
├── roadbuddy/ # Source Code Package 
│ ├── init.py 
│ ├── pipeline.py # Main Orchestrator (The "Brain") 
│ ├── perception/ # Module: Computer Vision 
│ │ ├── detector.py # YOLO wrapper (Ultralytics) 
│ │ ├── keyframe.py # Optical Flow / Similarity sampling 
│ │ └── transforms.py # Crop-and-zoom logic 
│ ├── rag/ # Module: Retrieval Augmented Generation 
│ │ ├── indexer.py # Builds indices from Law PDFs 
│ │ ├── retriever.py # Hybrid search (Dense + Sparse) 
│ │ └── schemas.py # Pydantic models for Law Articles 
│ ├── vlm/ # Module: Vision Language Model 
│ │ ├── client.py # vLLM OpenAI-compatible client 
│ │ └── prompt.py # Legal System Prompt templates 
│ └── utils/ 
│ ├── video.py # Fast decoding (Decord/OpenCV) 
│ └── timing.py # Context managers for 30s limit tracking 
├── scripts/ # Executable Scripts 
│ ├── build_knowledge_base.py # Pre-compute law embeddings 
│ ├── train_detector.py # Fine-tune YOLO on VNTS dataset 
│ ├── evaluate.py # Run local validation (Accuracy/F2) 
│ └── submission_infer.py # The script Zalo runs (main entry) 
└── tests/ # Unit & Integration Tests 
├── test_latency.py # CI check for <30s constraint 
└── test_legal_compliance.py # Verify law citations
```
### **3. Key Modules and Implementation Details**

**A. Perception Module (`roadbuddy/perception/`)**

This module solves the 'Small Object' problem. Standard VLMs resize images to $448 \times 448$, destroying distant sign details.

- `detector.py` $\rightarrow$ Wraps `ultralytics.YOLO`. Returns bounding boxes for signs.
- `keyframe.py` $\rightarrow$  Implements **Semantic Triggered Sampling** 
    - *Logic* $\rightarrow$ Instead of fixed FPS, calculate Optical Flow between frames. If flow magnitude < threshold (car stopped), drop frame. If YOLO confidence > 0.6 (sign detected), force keep frame.
    - *Goal* $\rightarrow$ Reduces 15s video (450 frames) to 6-8 highly informative frames.

**B. RAG Module (`roadbuddy/rag/`)**
This module ensures legal accuracy.
- `indexer.py`
    - Parses Law 36/2024 into chunks by *Article*.
    - Parses **QCVN 41** into chunks by *Sign ID* (e.g., P.101) + Image.
    - Uses **bge-m3** (multilingual) for text embeddings and **CLIP-ViT-L/14** for sign images embeddings.

- `retriever.py`
    - Performs **Hybrid Search** `(Vector_score * 0.7) + BM25_keyword_score * 0.3)` to balance semantic and exact matches.
    - *Why BM25?* If a question asks 'Article 9', vector search might miss it, but keyword search won't.

**C. VLM Module (`roadbuddy/vlm/`)**
This module handles the heavy reasoning.

- `client.py` $\rightarrow$  Instead of loading the model inside the python script (which is slow and memory intensive), it connects to a local vLLM server running in the background. This allows persistent KV-cache and continuous batching.
- `prompt.py` $\rightarrow$ Constructs the input
```
<|im_start|>system 
You are a Vietnamese Traffic Law expert. Answer based ONLY on the provided Context. Context Laws: {retrieved_law_text} Context Signs: {retrieved_sign_meanings} <|im_end|> 

<|im_start|>user 
Video Analysis: {yolo_detections_summary} Question: {user_question} <|im_end|> <|im_start|>assistant
```

### **4. Configuration (Hydra)**
`configs/config.yaml` is the main entry point. It allows easy swapping of components:
```yaml
defaults:
  - perception: yolov11_nano
  - vlm: qwen2.5_vl_awq
  - rag: qdrant_local

pipeline:
  max_latency: 28.0 # Leave 2s buffer
  debug_mode: false

submission:
  input_csv: "/data/public_test.csv"
  output_csv: "/data/submission.csv"
```

`configs/vlm/qwen2.5_vl_awq.yaml`
```yaml
model_name: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
backend: "vllm"
quantization: "awq"
dtype: "float16"
max_model_len: 8192 # Limit context to save VRAM
limit_mm_per_prompt:
  image: 8
  video: 1
```

### **5. Docker & Deployment Strategy**
The Zalo challenge requires a single Docker submission. We must start the vLLM server *inside* the container endpoint.

`docker/Dockerfile`
```Dockerfile
# Use vLLM official base image for optimized CUDA kernels
FROM vllm/vllm-openai:latest

# Install lightweight dependencies (YOLO, OpenCV, Qdrant)
RUN pip install --no-cache-dir \
    ultralytics==8.3.0 \
    opencv-python-headless \
    qdrant-client \
    hydra-core \
    decord

# Copy Model Weights (Assuming downloaded locally to./weights)
# This is crucial for offline inference
COPY./weights/Qwen2.5-VL-7B-AWQ /models/Qwen2.5-VL-7B-AWQ
COPY./weights/yolo11n.pt /models/yolo11n.pt
COPY./weights/bge-m3 /models/bge-m3

# Copy Source Code
COPY. /app
WORKDIR /app

# Set Entrypoint
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
```

`docker/entrypoint.sh`
```bash
#!/bin/bash

# 1. Start vLLM Server in background (Daemon)
# This serves the heavy model once, avoiding reload overhead per sample
python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen2.5-VL-7B-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port 8000 &

# 2. Wait for server to be ready
echo "Waiting for vLLM to start..."
while! curl -s http://localhost:8000/health > /dev/null; do sleep 1; done
echo "vLLM Ready!"

# 3. Run the Zalo submission script
python3 scripts/submission_infer.py

# 4. Kill vLLM after finished
pkill -f vllm
```

### **6. Adaptation from NVIDIA Nemo/Skills**
We adapted the **Skills** philosophy of separating *Skills* (capabilities) from *Evaluation*:

1. **Recipe-based workflow** $\rightarrow$ Just like Nemo uses recipes for training LLMs, we use Hydra configs as 'recipes' for inference pipelines. You can have a `recipe_speed.yaml` (YOLOv11n + MiniCPM) and `recipe_accuracy.yaml` (YOLOv11x + Qwen2.5-VL) for different trade-offs.

2. **Cluster-ready** $\rightarrow$ The `scripts/` folder is designed to run locally on Docker or be dispatched to a Slurm cluster (if you were training), but optimized here for the single-node inference constraint.

3. **Evaluation harness** $\rightarrow$ The `tests/test_latency.py` module mimics Nemo's rigorous testing. It fails the build if any single sample takes >29.5 seconds, ensuring you do not score 0 in the competition due to timeouts.

### **7. Execution**
**To run locally (Simulating Submission)**
```bash
# 1. Build Docker
docker build -t roadbuddy:v1 -f docker/Dockerfile.

# 2. Run with GPU
docker run --gpus all \
    -v $(pwd)/data:/data \
    roadbuddy:v1
```

This repository structure provides the **rigidty** needed for a competition (wont break easily) with the **flexibility** needed for research (easy to swap new models next week).