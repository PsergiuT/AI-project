# Anterior Ethmoidal Artery Segmentation on CBCT
## Theory & Background Documentation

**Project:** AEA Segmentation on CBCT using a Vision Transformer Agent  
**Author:** Sergiu Pop  
**Date:** May 2026

---

## Table of Contents

1. [Part I — Medical & Biological Background](#part-i)
   - 1.1 Anatomy of the Anterior Ethmoidal Artery
   - 1.2 Anatomical Variations of the AEA
   - 1.3 Cone Beam Computed Tomography (CBCT) in ENT
   - 1.4 Endoscopic Sinus Surgery and the Risk of AEA Injury
   - 1.5 Current Approaches to AEA Localization
2. [Part II — Artificial Intelligence Background](#part-ii)
   - 2.1 Medical Image Segmentation
   - 2.2 From Convolutional Networks to Vision Transformers
   - 2.3 The Transformer and Self-Attention Mechanism
   - 2.4 The Swin Transformer
   - 2.5 SwinUNETR — Architecture Deep Dive
   - 2.6 Transfer Learning and Fine-Tuning
   - 2.7 Training Pipeline and Class Imbalance
   - 2.8 LLM-Based Agent Architecture
   - 2.9 Evaluation Metrics for Segmentation

---

<a name="part-i"></a>
# Part I — Medical & Biological Background

---

## 1.1 Anatomy of the Anterior Ethmoidal Artery

The anterior ethmoidal artery (AEA) is a branch of the ophthalmic artery, which itself is the first intracranial branch of the internal carotid artery. The ophthalmic artery enters the orbit through the optic canal and gives rise to several branches, among which the anterior and posterior ethmoidal arteries are of particular clinical significance for surgeons operating in the sinonasal region.

The AEA exits the orbit through the anterior ethmoidal foramen, located on the medial orbital wall (the lamina papyracea), and traverses the anterior ethmoid air cells in a bony canal or, in a clinically important subset of patients, courses through the ethmoid air cells suspended in a mesentery without bony protection — a configuration that dramatically increases the risk of inadvertent surgical injury. After crossing the ethmoid labyrinth, the artery enters the anterior cranial fossa through a small canal in the cribriform plate, where it gives a branch to the dura mater before descending through the nasal slit and terminating as the external nasal artery on the dorsum of the nose.

The artery supplies the anterior ethmoid air cells, the dura of the anterior cranial fossa, the upper part of the nasal septum and lateral wall, and contributes to the blood supply of the external nasal dorsum. Despite its small caliber — typically 1 to 2 millimeters in diameter — the AEA is under significant arterial pressure as a direct branch of the ophthalmic artery, meaning even minor injury can produce rapid and expansive hemorrhage into the orbit.

The spatial relationship between the AEA and the ethmoid roof (also called the fovea ethmoidalis or skull base) is the key anatomical landmark for surgeons. In most subjects the artery runs along or just below the skull base, but the distance from the artery to the basal lamella, the degree of pneumatization of the surrounding air cells, and the height of the ethmoid roof are all subject to significant inter-individual variation, making preoperative imaging indispensable.

---

## 1.2 Anatomical Variations of the AEA

One of the central challenges in endoscopic sinus surgery is the unpredictable anatomical course of the AEA. Amarnath & Suresh Kumar (2019) conducted a systematic study of AEA variants on CT of paranasal sinuses, establishing a classification framework that directly informs the need for automated preoperative localization tools.

The key variants described in the literature are:

**Type I (Most common, ~75–80% of cases):** The AEA runs within a bony canal that is an integral part of the skull base (fovea ethmoidalis). In this configuration the artery is reasonably protected and can be identified as a bony ridge or groove on CT.

**Type II (10–15% of cases):** The artery runs partially suspended below the skull base within the ethmoid air cells, attached by a thin mesentery of bone. This variant increases surgical risk because the artery is more exposed and can be inadvertently torn if the surgeon is unaware of its position.

**Type III (5–8% of cases):** The AEA hangs freely within the ethmoid air cells with minimal or no bony protection. This is the highest-risk configuration; injury in this setting frequently results in arterial retraction into the orbit and rapid formation of an orbital hematoma.

Beyond the vertical position relative to the skull base, additional variations include the distance from the anterior wall of the sphenoid sinus, the relationship to the basal lamella of the middle turbinate, bilaterally asymmetric courses (the left and right AEA may follow entirely different variants in the same patient), and the rare but documented absence of one or both arteries.

These variations make manual pre-operative localization error-prone and time-consuming, and motivate the development of automated segmentation systems. Huang et al. (2020) demonstrated that even experienced radiologists show inter-observer variability in AEA localization on CT, and that an AI-assisted system can reduce this variability while matching or exceeding human accuracy.

---

## 1.3 Cone Beam Computed Tomography (CBCT) in ENT

Cone Beam Computed Tomography (CBCT) is a specialized imaging modality derived from conventional CT but optimized for high-resolution imaging of bony and mineralized structures at a lower radiation dose. Instead of a fan-shaped X-ray beam rotating around the patient as in conventional multi-detector CT (MDCT), CBCT uses a cone-shaped beam that captures a full 3D volume in a single rotation of the source-detector pair around the patient's head.

The key advantages of CBCT over conventional CT in the context of ENT and paranasal sinus imaging are:

**Spatial resolution:** CBCT achieves isotropic voxel sizes of 0.2–0.4mm, compared to 0.5–1mm for conventional CT. This is critical for resolving small structures like the AEA canal, which may be only 1–2mm in diameter. The dataset used in this project uses 0.4mm isotropic voxels, which sits at the upper limit of resolution for this modality.

**Radiation dose:** CBCT delivers significantly lower radiation than conventional CT (effective dose of 20–200 μSv vs. 200–2000 μSv), which is important when imaging patients for pre-operative planning.

**Cost and accessibility:** CBCT scanners are smaller and less expensive than conventional CT systems, making them more widely available in ENT outpatient settings.

**Limitations:** CBCT has lower soft tissue contrast compared to conventional CT or MRI. Since the AEA is a soft tissue structure (a blood vessel), it is not directly visible on CBCT — what is visible is the bony canal or groove that the artery passes through, and the surrounding air cells and bone. This means segmentation on CBCT is inherently a task of localizing the anatomical corridor of the artery rather than the vessel wall itself, which is an important distinction when interpreting the ground truth annotations in this dataset.

---

## 1.4 Endoscopic Sinus Surgery and the Risk of AEA Injury

Functional Endoscopic Sinus Surgery (FESS) is the standard surgical approach for chronic rhinosinusitis, nasal polyposis, benign sinonasal tumors, and skull base access. The procedure is performed entirely through the nostrils using rigid endoscopes and small powered instruments, without any external incisions. The surgeon navigates through a complex three-dimensional anatomical space — the paranasal sinuses — guided by endoscopic vision and, increasingly, by intraoperative image guidance systems.

The AEA represents one of the highest-risk anatomical structures in FESS for two reasons. First, as described above, its course is variable and unpredictable without preoperative imaging. Second, injury to the AEA within the ethmoid air cells can cause the arterial stump to retract into the orbit — because the artery crosses the medial orbital wall — producing a rapidly expanding retrobulbar hematoma. If not recognized and decompressed within minutes, this complication can lead to ischemic optic neuropathy and permanent visual loss. The reported incidence of AEA injury during FESS ranges from 0.3% to 1% in large series, which translates to hundreds of patients annually at high-volume centers.

Itayem et al. (2019) demonstrated in a prospective clinical study that preoperative identification of the AEA using segmented image guidance — essentially providing the surgeon with a 3D map of the artery's course before and during surgery — significantly increased surgeon confidence, localization accuracy, and operative efficiency. The transition from manual segmentation (which takes 15–30 minutes per case by a trained radiologist) to automated AI-based segmentation is therefore not merely a computational convenience but has direct clinical impact: it removes the manual bottleneck, standardizes the quality of preoperative assessment, and makes this level of surgical preparation economically viable at scale.

---

## 1.5 Current Approaches to AEA Localization

Prior to the application of deep learning, AEA localization on CT relied on manual review by a radiologist or otolaryngologist, with identification of the bony canal using axial, coronal, and sagittal reconstructions. Semi-automated approaches included region-growing algorithms and threshold-based bony canal detection, but these required significant manual initialization and were sensitive to imaging artifacts.

Huang et al. (2020) were among the first groups to report a dedicated AI algorithm for AEA localization, using a convolutional neural network trained on sinus CT scans with radiologist-annotated AEA positions. Their system achieved performance comparable to expert radiologists on a held-out test set, establishing the feasibility of automated AI-based localization. However, their approach was limited to detection (predicting a bounding box or point location) rather than full volumetric segmentation, which provides less information for surgical planning.

The present project extends this work by performing full 3D segmentation — delineating the complete course of both the left and right AEA through the ethmoid sinuses — using a more powerful Vision Transformer architecture and a dataset of 130 manually annotated CBCT scans.

---

<a name="part-ii"></a>
# Part II — Artificial Intelligence Background

---

## 2.1 Medical Image Segmentation

Image segmentation is the task of assigning a label to every pixel (in 2D) or voxel (in 3D) of an image, partitioning it into meaningful regions. In the context of medical imaging, segmentation means identifying which voxels belong to a specific anatomical structure — in our case, the anterior ethmoidal artery — and which belong to the background or other tissues.

Segmentation is distinct from classification (which assigns a single label to the entire image) and from detection (which identifies a bounding box around a structure). It provides the most spatially precise information, which is why it is the preferred output format for surgical planning tools.

The difficulty of medical image segmentation stems from several factors: high dimensionality (3D volumes with millions of voxels), class imbalance (a small structure like the AEA occupies a tiny fraction of the total volume), inter-subject anatomical variability, imaging artifacts, and the cost of obtaining ground truth annotations (which require expert clinicians and many hours of manual work).

Early approaches to medical segmentation used classical computer vision techniques — thresholding, region growing, graph cuts, and atlas-based registration. These methods required extensive manual parameter tuning and did not generalize well across patients. The advent of deep learning, and specifically the U-Net architecture in 2015, marked a turning point.

---

## 2.2 From Convolutional Networks to Vision Transformers

**U-Net (2015)** was the first deep learning architecture purpose-built for biomedical image segmentation. It introduced the encoder-decoder structure with skip connections: the encoder progressively reduces spatial resolution while extracting features, the decoder progressively restores spatial resolution, and skip connections pass feature maps directly from encoder to decoder to preserve fine spatial detail. U-Net became the dominant architecture in medical imaging for the following decade.

**3D U-Net** extended this to volumetric segmentation by replacing 2D convolutions with 3D convolutions, allowing the model to learn spatial context across slices — essential for structures like the AEA that follow a curved 3D trajectory through the sinuses.

**Convolutional Neural Networks (CNNs)**, which power U-Net and its variants, process images by applying learned filters (kernels) that detect local patterns — edges, textures, shapes. Their strength is local feature extraction, but their weakness is limited long-range context: a convolutional filter can only see a small neighborhood at a time, and capturing relationships between distant parts of the image requires many stacked layers.

**Vision Transformers (ViT, 2020)** introduced an entirely different approach borrowed from Natural Language Processing (NLP). Instead of sliding filters across the image, a ViT divides the image into fixed-size patches (analogous to words in a sentence) and processes them using the self-attention mechanism — allowing every patch to directly interact with every other patch regardless of spatial distance. This gives ViTs a global receptive field from the very first layer, which is particularly valuable for capturing the full trajectory of a structure like the AEA that spans a large portion of the imaging volume.

---

## 2.3 The Transformer and Self-Attention Mechanism

The core operation of the Transformer is **self-attention**, which can be understood intuitively as follows. Given a sequence of input patches, for each patch the model asks: "which other patches in this image are most relevant to understanding what I am?" It then computes a weighted sum of all other patches, where the weights (called attention scores) reflect how relevant each patch is. This process is computed in parallel for all patches simultaneously.

Formally, each patch is projected into three vectors — a Query (Q), a Key (K), and a Value (V). The attention score between patch i and patch j is computed as the dot product of Q_i and K_j, normalized by the square root of the feature dimension, and passed through a softmax to produce a probability distribution. The output for patch i is then the weighted sum of all Value vectors V_j, weighted by the attention scores.

This mechanism allows the model to learn that, for example, a patch containing the ethmoid roof is highly relevant to a patch containing the AEA canal — even if they are spatially distant — because they co-occur consistently in the training data. CNNs would require many stacked layers to learn this relationship indirectly; Transformers capture it directly.

**Multi-head attention** runs this process in parallel with multiple different sets of Q, K, V projections (called heads), allowing the model to simultaneously attend to multiple different types of relationships. The outputs of all heads are concatenated and projected back to the original feature dimension.

The full Transformer block wraps multi-head attention with residual connections (which help gradients flow during training) and layer normalization (which stabilizes training), followed by a small feedforward network applied independently to each patch.

---

## 2.4 The Swin Transformer

The original Vision Transformer, applied directly to medical volumes, has a critical computational limitation: the self-attention operation scales quadratically with the number of patches. A 3D volume of 400×400×250 voxels divided into 4×4×4 patches produces tens of thousands of patches, making global self-attention computationally prohibitive.

The **Swin Transformer** (Shifted Window Transformer, Liu et al., 2021) solves this by computing self-attention within local windows rather than globally, and then shifting the window partition between layers to allow cross-window interaction. This reduces the computational complexity from quadratic to linear in the number of patches, making it practical for high-resolution 3D medical volumes.

The Swin Transformer also introduces a hierarchical architecture: as depth increases, patches are merged together (like pooling in a CNN), progressively increasing the receptive field while decreasing spatial resolution. This creates feature maps at multiple scales — a property that is essential for the encoder-decoder architecture used in segmentation.

---

## 2.5 SwinUNETR — Architecture Deep Dive

**SwinUNETR** (Swin UNEt TRansformer, Tang et al., 2022) is the architecture used in this project. It combines the Swin Transformer as an encoder with a CNN-based decoder in the U-Net style, creating a hybrid architecture that benefits from both the global context of Transformers and the spatial precision of convolutional decoders.

The architecture proceeds as follows:

**Input:** A 3D patch of size 96×96×96 voxels is extracted from the CBCT volume and fed into the network.

**Patch Embedding:** The 3D patch is divided into non-overlapping tokens of size 2×2×2 voxels. Each token is linearly projected to a feature vector of dimension 48, producing a sequence of 48×48×48 tokens.

**Swin Transformer Encoder (4 stages):** The token sequence passes through four stages of Swin Transformer blocks with progressively merged patches:
- Stage 1: 48×48×48 tokens, feature dimension 48
- Stage 2: 24×24×24 tokens, feature dimension 96
- Stage 3: 12×12×12 tokens, feature dimension 192
- Stage 4: 6×6×6 tokens, feature dimension 384

Each stage uses multiple Swin Transformer blocks with alternating regular and shifted window attention, allowing information to flow both within and across local windows.

**Skip Connections:** Feature maps from each encoder stage are passed directly to the corresponding decoder stage, preserving fine-grained spatial information that would otherwise be lost during downsampling.

**CNN Decoder (4 stages):** The decoder uses transposed convolutions (also called deconvolutions) to progressively upsample the feature maps back to the original resolution, concatenating skip connections from the encoder at each stage. Each decoder stage applies residual convolutional blocks to refine the features.

**Output Head:** A 1×1×1 convolution maps the final feature map to 3 channels (background, AEA Left, AEA Right), followed by a softmax activation that produces per-voxel class probabilities.

The total parameter count of SwinUNETR (base configuration) is approximately 62 million parameters, making it a large but tractable model for fine-tuning on a dataset of 130 cases when pre-trained weights are used as a starting point.

---

## 2.6 Transfer Learning and Fine-Tuning

**Transfer learning** is the practice of initializing a model with weights learned on a large dataset (pre-training) before adapting it to a specific task on a smaller dataset (fine-tuning). It is the primary technique that makes deep learning viable for medical imaging tasks with limited annotated data.

SwinUNETR was pre-trained by Tang et al. (2022) using a self-supervised learning objective called **masked volume inpainting**: random patches of the input volume are masked out and the model is trained to reconstruct them. This forces the encoder to learn rich, generalizable representations of 3D anatomical structures without requiring any segmentation labels. The pre-training was conducted on a large collection of unlabeled CT and MRI volumes.

When we fine-tune on our 130 AEA CBCT cases, the pre-trained encoder already understands 3D anatomical structure — bone density patterns, air-tissue boundaries, vascular corridors — and only needs to specialize this knowledge to the specific task of AEA localization. This is analogous to teaching a medical student who already has a strong foundation in anatomy to specialize in otolaryngology, rather than teaching anatomy from scratch.

In our fine-tuning configuration, all layers of the network are unfrozen and allowed to update during training (full fine-tuning), but the learning rate is kept small (1e-4) to prevent overwriting the pre-trained representations too aggressively. The MONAI model zoo provides the pre-trained SwinUNETR weights in a format directly compatible with the fine-tuning pipeline.

---

## 2.7 Training Pipeline and Class Imbalance

**Patch-based training** is necessary because modern GPUs cannot hold an entire 400×400×250 volume in memory simultaneously during training. Instead, random patches of size 96×96×96 are extracted from the volume and fed to the network. During inference, the full volume is processed using a sliding window approach with overlap, and the predictions from overlapping patches are averaged.

**Foreground oversampling** addresses the class imbalance problem. In a volume where the AEA occupies fewer than 0.1% of all voxels, randomly sampled patches would rarely contain any AEA voxels, giving the model almost no signal about what it actually needs to learn. By forcing 50% of patches to be centered on or near an AEA voxel (foreground sampling) and 50% to be sampled randomly (background sampling), the model sees the target structure in every other training step.

**Data augmentation** artificially expands the training set by applying random transformations that preserve the clinical validity of the data. The augmentations used in our pipeline include: random flips along all three axes (the AEA can be on either side), random rotations up to 15 degrees, random scaling between 0.85× and 1.15×, and random intensity shifts and scaling to simulate scanner variability. These transformations prevent the model from memorizing the exact appearance of training cases and improve generalization.

**DiceCELoss** is the composite loss function used for training. It combines two complementary signals:

The *Dice Loss* component measures the overlap between the predicted and ground truth segmentation. It is defined as:

```
Dice Loss = 1 - (2 × |Pred ∩ GT|) / (|Pred| + |GT|)
```

where |Pred| and |GT| are the volumes of the prediction and ground truth respectively. Dice Loss is naturally insensitive to class imbalance because it normalizes by the size of the structures being compared — a perfect prediction of a small structure scores 0 loss just as a perfect prediction of a large structure does.

The *Cross Entropy Loss* component measures the per-voxel prediction confidence and stabilizes training in the early epochs when the Dice Loss gradient can be noisy.

The **AdamW optimizer** with **cosine annealing learning rate schedule** is used to update the network weights. AdamW adapts the learning rate for each parameter individually based on gradient history, making it robust to the highly variable gradient magnitudes that arise in medical segmentation tasks. Cosine annealing smoothly reduces the learning rate from its initial value to near zero over the training run, allowing the model to make rapid initial progress and then fine-tune its weights precisely near convergence.

---

## 2.8 LLM-Based Agent Architecture

The segmentation model alone produces outputs but cannot interpret user intent, adapt to varied inputs, or explain its reasoning. To satisfy the requirement for an autonomous AI agent, the project wraps the segmentation pipeline in a **Large Language Model (LLM) agent** that can interpret natural language instructions and orchestrate the pipeline end-to-end.

**What is an LLM agent?** A standard LLM (such as Llama 3.1) produces text given a text prompt. An LLM agent extends this by giving the LLM access to a set of callable tools — Python functions that interact with the real world (loading files, running models, computing metrics). The LLM is prompted to reason about the user's request, decide which tools to call and in what order, interpret the results, and produce a final answer.

**The ReAct pattern** (Reasoning + Acting) is the standard framework for LLM agents. In each step the agent produces a Thought (internal reasoning about what to do next), then an Action (a tool call with specific arguments), then observes the Action result (tool output), and repeats until it has enough information to produce a Final Answer. This creates a transparent, interpretable reasoning trace that demonstrates the agent's decision-making process.

**LangChain** is the Python framework used to implement the agent. It provides abstractions for defining tools (as decorated Python functions), connecting them to an LLM backend, and running the ReAct loop. Each tool in our pipeline is defined as a LangChain `@tool` function with a natural language description that helps the LLM understand when and how to use it.

**Ollama** is used to run the Llama 3.1 8B model locally without any cloud dependency or API cost. It provides a lightweight inference server that exposes the model through a standard HTTP API compatible with LangChain's Ollama integration. While smaller than cloud-hosted models, Llama 3.1 8B has demonstrated adequate function-calling capability for structured, well-defined tool orchestration pipelines.

The five tools available to the agent are:

1. `load_and_preprocess(dicom_path: str)` — Loads a DICOM series from the given path, converts it to a normalized 3D tensor, and returns a volume object ready for inference.
2. `run_segmentation(volume)` — Runs the fine-tuned SwinUNETR model on the volume using sliding window inference and returns the 3-class segmentation mask.
3. `postprocess(mask)` — Applies connected component analysis to remove small spurious predictions (false positives) and returns a cleaned mask.
4. `evaluate(pred_mask, gt_mask)` — Computes Dice Score, IoU, and HD95 for both AEAL and AEAR and returns a structured metrics dictionary.
5. `generate_report(metrics, patient_id)` — Formats the metrics and key findings into a structured JSON report that is displayed in the web UI and available for download.

The Gradio web interface provides the entry point: the user uploads a CBCT scan and types a natural language instruction. The instruction and file path are passed to the agent, which runs the ReAct loop to completion and returns the segmentation and report to the UI.

---

## 2.9 Evaluation Metrics for Segmentation

Three complementary metrics are reported to assess segmentation quality:

**Dice Similarity Coefficient (DSC)** measures volumetric overlap between the predicted mask (P) and the ground truth mask (G):

```
DSC = (2 × |P ∩ G|) / (|P| + |G|)
```

DSC ranges from 0 (no overlap) to 1 (perfect overlap). For small vessel segmentation tasks comparable to AEA, a DSC above 0.70 is generally considered clinically useful, with values above 0.80 considered strong performance. DSC is reported separately for AEAL and AEAR.

**Intersection over Union (IoU / Jaccard Index)** is a closely related metric:

```
IoU = |P ∩ G| / |P ∪ G|
```

IoU is always lower than DSC for the same prediction (it penalizes false positives and false negatives more strongly). The relationship between them is: IoU = DSC / (2 - DSC). Reporting both is standard practice in the medical segmentation literature and provides a complete picture of overlap quality.

**Hausdorff Distance at 95th percentile (HD95)** measures boundary accuracy rather than volumetric overlap. It is defined as the 95th percentile of the set of distances from each voxel on the predicted boundary to the nearest voxel on the ground truth boundary (and vice versa). Unlike the maximum Hausdorff Distance, the 95th percentile variant ignores the most extreme outlier boundary errors, making it more robust to small isolated prediction errors.

HD95 is measured in millimeters (our dataset uses 0.4mm voxels). A lower HD95 indicates that the predicted boundary closely follows the true AEA boundary. For clinical use in surgical planning, an HD95 below 2mm is generally considered acceptable — meaning the predicted artery course is within 2mm of the true course at the 95th percentile of boundary points.

Together, DSC and IoU measure how much of the structure is correctly identified, while HD95 measures how precisely the boundaries are localized. Both dimensions of performance are clinically relevant: DSC ensures the artery is not missed, and HD95 ensures the predicted course is accurate enough to guide surgical planning.

---

## References

1. Huang, J., Habib, A. R., Mendis, D., Chong, J., Smith, M., Duvnjak, M., ... & Wong, E. (2020). An artificial intelligence algorithm that differentiates anterior ethmoidal artery location on sinus computed tomography scans. *The Journal of Laryngology & Otology*, 134(1), 52–55.

2. Amarnath, S., & Suresh Kumar, P. (2019). Study of variants of anterior ethmoidal artery on computed tomography of paranasal sinuses. *Int J Otorhinolaryngol Head Neck Surg*, 5, 19–23.

3. Itayem, D. A., Anzalone, C. L., White, J. R., Pallanch, J. F., & O'Brien, E. K. (2019). Increased accuracy, confidence, and efficiency in anterior ethmoidal artery identification with segmented image guidance. *Otolaryngology–Head and Neck Surgery*, 160(5), 818–821.

4. Tang, Y., Yang, D., Li, W., Roth, H. R., Landman, B., Xu, D., ... & Nath, V. (2022). Self-supervised pre-training of swin transformers for 3D medical image analysis. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 20730–20740.

5. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 10012–10022.

6. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 234–241.

---

*End of Theory Documentation — Phase 3*
