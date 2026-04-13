# Multi-Modal Interrogation of Visual Cortical Circuits: Linking Transcriptomic Identity, Spatial Organization, and Neural Dynamics in Mouse V1

---

## 1. Project and Dataset Description

### 1.1 Overview

This project investigates the relationship between transcriptomic cell-type identity, spatial organization, functional neural dynamics, and behavioral state in the mouse primary visual cortex (V1). It combines large-scale two-photon calcium imaging with post-hoc spatial transcriptomics (10x Xenium) and hierarchical cell-type classification (Allen Institute for Brain Science taxonomy), enabling a uniquely integrated analysis of how genetically defined cell types encode visual information, interact within local circuits, and adapt their responses to changing stimulus contexts and behavioral states.

### 1.2 Dataset Summary

| Feature | Details |
|---|---|
| **Species / Brain Region** | Mouse, primary visual cortex (VISp), layers 1–4/5 |
| **Imaging Modality** | Two-photon calcium imaging (GCaMP, pan-neuronal), ΔF/F extracted at 10 Hz |
| **Imaging Depths** | 8 planes from cortical surface: 40, 80, 120, 160, 200, 220, 280, 320 µm |
| **Field of View** | 400 × 400 µm |
| **Animals** | 3 mice in current grating dataset (IDs: 778174, 786297, 797371); 4 mice total across all experiments |
| **Cells per Mouse** | ~600–1,200 simultaneously recorded neurons per mouse |
| **Total Cells (Grating Sessions)** | 2,824 neurons across 3 mice |
| **Cell Types Recorded** | Excitatory (L2/3 IT, L4/5 IT, L5 ET) and inhibitory (Pvalb, Sst, Vip, Lamp5) |
| **Transcriptomics** | 299 genes per cell via 10x Xenium standard panel (post-hoc, matched to in vivo cells) |
| **Cell-Type Taxonomy** | AIBS hierarchical: class → subclass → supertype → cluster (50–72 clusters, 22–28 supertypes, 6–7 subclasses, 3 classes per mouse); bootstrapping probabilities at each level |
| **Spatial Information** | 3D coordinates (x, y, z) for each cell |
| **Morphology** | Cell-body mask properties: 3D centroid, volume (n_voxels), size along 3 PCA axes (size_pc1/2/3_um), bounding-box dimensions (size_x/y/z_um), soma orientation angle (angle_deg_xy) |
| **Behavioral Monitoring** | Running speed (continuous), with trial-level running/stationary classification |
| **Pre-fitted GLM** | Ridge-regularized encoding model per cell per session: visual, state (running/pupil), and combined components; y, y_hat, y_hat_visual, y_hat_state; per-condition R² scores |
| **Spontaneous Activity** | ~360 s grey-screen epochs (3,624 timepoints at 10 Hz) recorded before/after stimulus blocks, per session |
| **Catch Trials** | 27 blank (grey-screen) stimulus presentations per session (51 timepoints, −1 to +4 s), with running data |

### 1.3 Experimental Design

**Four experimental sessions per mouse (passive viewing, separate days):**

| Session | Stimulus | Details |
|---|---|---|
| **Exp 1** | Natural movie clips | ~1,000 unique clips (2–5 sec each); ~20 clips repeated 20 times for reliability estimation |
| **Exp 2–4** | Full-field drifting gratings | 8 directions × 5 contrasts (0.05, 0.1, 0.2, 0.4, 0.8) × 5 temporal frequencies (1, 2, 4, 8, 15 Hz); spatial frequency fixed at 0.04 cpd; 2-second trials |

**Context/Block Structure (Grating Sessions):**

Each grating session contains 4 interleaved blocks that define two distinct **stimulus contexts**:

| Block | Context Type | Constant Parameter | Varying Parameters |
|---|---|---|---|
| Blocks 0, 2 | **Contrast context** | Temporal frequency = 1 Hz | Contrast (5 levels) × Direction (8) |
| Blocks 1, 3 | **Speed context** | Contrast = 0.8 | Temporal frequency (5 levels) × Direction (8) |

Each block contains ~800–825 stimulus presentations. The same grating session is repeated across 3 days (dates), enabling analysis of adaptation across sessions.

### 1.4 Data Structure

#### Zarr Multimodal Stores (`.zarr`)

Each `{mouse_id}_multimodal_data.zarr` combines all modalities in a single hierarchical store with four top-level groups:

- **`unique_id`** (n_cells,): Cell IDs matching across modalities
- **`morphology/mask_properties/`**: Cell-body segmentation mask features (n_cells each):
  - `centroid_x_um`, `centroid_y_um`, `centroid_z_um`: 3D spatial coordinates in µm
  - `n_voxels`: Cell body volume proxy (range 253–8,337 across mice)
  - `size_pc1_um`, `size_pc2_um`, `size_pc3_um`: Principal axes of cell body shape (soma elongation)
  - `size_x_um`, `size_y_um`, `size_z_um`: Bounding-box dimensions
  - `angle_deg_xy`: Orientation angle of cell body in the XY imaging plane (−180° to +180°)
- **`transcriptomics/`**:
  - `cell_type/`: 13 arrays — hierarchical labels (class/subclass/supertype/cluster) with `_name`, `_label`, and `_bootstrapping_probability` at each level, plus `cluster_alias`
  - `cellxgene/`: 299 gene expression arrays (float64, one per gene)
- **`ophys/drifting_gratings/{session_1,session_2,session_3}/`**: Each session contains:
  - **`stim_aligned_dff/GratingStim/`**: Trial-resolved calcium activity
    - `dff`: (2186 trials, 41 timepoints, n_cells) — ΔF/F at 10 Hz, −1 s to +3 s
    - `running`: (2186, 41, 2) — speed and acceleration per timepoint
    - `time_relative`: (41,) — time axis from −1.0 to +3.0 s
    - `trial_info/`: 13 stimulus metadata arrays (contrast, orientation, temporal_frequency, spatial_frequency, stim_block, stim_name, gray_screen, start/stop_time, duration, stim_index, stim_index_block, stim_type)
  - **`stim_aligned_dff/Catch/`**: Catch (blank) trials
    - `dff`: (27 trials, 51 timepoints, n_cells) — from −1 s to +4 s
    - `running`: (27, 51, 2); `time_relative`: (51,)
    - `trial_info/`: same structure (all stimulus fields are NaN; `gray_screen=True`)
  - **`stim_aligned_dff/GreyScreen/`**: Spontaneous activity epochs
    - `dff`: (2 epochs, 3624 timepoints, n_cells) — ~19 s and ~360 s grey-screen periods at 10 Hz
    - `running`: (2, 3624, 2); `time_relative`: (3624,) spanning ~362 s
    - `trial_info/`: `stim_name='spontaneous'`, `gray_screen=True`
  - **`glm/`**: Pre-fitted ridge-regularized GLM encoding model
    - `alpha/alphas`: (n_glm_cells,) — selected regularization strength per cell (values: 10, 100, or 1000)
    - `coef/`: 167 coefficient arrays, each (n_glm_cells, n_basis) — temporal basis weights per stimulus condition, including `coef_visual` (n_glm_cells, 4920), `coef_state` (n_glm_cells, 21), `coef_pupil_running` (n_glm_cells, 21), and per-condition kernels (30 basis functions each)
    - `score/`: 171 score arrays — per-condition R² values, plus `score_total`, `score_train`, `score_test`, `score_eval`, `score_visual`, `score_state`, `score_pupil_running`
    - `y/y`: (n_glm_cells, ~37,900) — observed ΔF/F (continuous, not trial-segmented)
    - `y_hat/`: `y_hat` (full prediction), `y_hat_visual` (visual-only prediction), `y_hat_state` (state-only prediction)
    - **Note:** GLM is fit on a subset of cells (e.g., 158/614 for mouse 778174; all 1,173 for mouse 786297)
  - **`tuning_properties/`**: Precomputed per-cell tuning metrics (12 float64 arrays, n_cells each):
    - `OSI`, `DSI`, `pref_ori`, `max_response`, `mean_response`, `bandwidth`, `C50`, `Rmax_crf`, `n_exponent`, `pref_TF`, `TF_max_response`, `TF_lowpass_idx`

### 1.5 Cell-Type Composition

| Subclass | Mouse 778174 | Mouse 786297 | Mouse 797371 | Approx. % |
|---|---|---|---|---|
| L4/5 IT CTX Glut | 313 | 549 | 498 | ~48% |
| L2/3 IT CTX Glut | 245 | 510 | 413 | ~41% |
| Pvalb GABAergic | 26 | 49 | 45 | ~4% |
| Vip GABAergic | 20 | 26 | 29 | ~3% |
| Sst GABAergic | 6 | 30 | 20 | ~2% |
| Lamp5 GABAergic | 4 | 9 | 9 | ~1% |
| L5 ET CTX Glut | — | — | 23 | <1% |

---

## 2. Scientific Questions, Hypotheses, and Analysis Plans

Questions are organized into **six thematic domains** (Broad → Specific), each with testable hypotheses and detailed analysis plans.

---

### DOMAIN A: Transcriptomic Identity and Neural Coding

**Broad Question A:** *How does transcriptomic cell-type identity relate to the functional properties of individual neurons and population-level coding in V1?*

---

#### A1. Do transcriptomically defined cell types have distinct tuning properties?

**Hypothesis A1:** Neurons within the same transcriptomic subclass (e.g., L4/5 IT, L2/3 IT, Pvalb, Sst, Vip, Lamp5) share more similar orientation selectivity, contrast response functions, and temporal frequency tuning than neurons across subclasses, but significant functional heterogeneity exists within subclasses that maps onto finer taxonomic levels (supertype, cluster).

**Analysis Plan:**
1. **Compute tuning curves per cell:**
   - **Orientation tuning:** For each cell, average ΔF/F across trials for each of the 8 directions. Fit a double-Gaussian (von Mises) model. Extract: preferred orientation, orientation selectivity index (OSI), direction selectivity index (DSI), tuning bandwidth.
   - **Contrast response function (CRF):** Average responses across the 5 contrast levels (within blocks 0, 2 at TF=1 Hz). Fit Naka-Rushton function: extract C50 (semi-saturation contrast), Rmax, baseline, exponent n.
   - **Temporal frequency tuning:** Average responses across 5 TF levels (blocks 1, 3 at contrast=0.8). Fit log-Gaussian; extract preferred TF, bandwidth.
2. **Compare tuning parameters across cell types:**
   - ANOVA / Kruskal-Wallis tests across subclasses for each tuning parameter.
   - Post-hoc pairwise comparisons (e.g., Dunn's test) between specific subclass pairs.
   - Repeat at supertype and cluster levels to assess if finer taxonomy explains more variance.
3. **Variance decomposition:**
   - Nested ANOVA: variance in tuning parameters explained by class → subclass → supertype → cluster.
   - Compare to shuffled cell-type labels (permutation test) to assess significance.
4. **Visualization:**
   - Population tuning curve plots (mean ± SEM per subclass), overlaid.
   - Violin/swarm plots of OSI, DSI, C50, preferred TF, split by subclass.
   - Hierarchical clustering dendrogram of cells based on tuning vectors, colored by transcriptomic label.
   - Confusion matrix: can you predict subclass from tuning parameters (logistic regression / random forest)?

---

#### A2. Does gene expression predict functional response properties at the single-cell level?

**Hypothesis A2:** Specific genes (beyond canonical markers like Pvalb, Sst, Vip) predict quantitative aspects of neural tuning, particularly ion channels (Kcnh5, Cacna2d2), neuropeptides (Cck, Npy, Penk), and synaptic molecules (Syt2, Cbln4, Nrn1).

**Analysis Plan:**
1. **Gene-tuning correlation analysis:**
   - For each of the ~250 genes, correlate expression level with each tuning metric (OSI, DSI, C50, preferred TF, response amplitude, reliability) using Spearman rank correlation.
   - Correct for multiple comparisons (Benjamini-Hochberg FDR).
   - Within-subclass analysis: repeat correlations within each subclass to separate cell-type effects from gene-expression effects.
2. **Regularized regression:**
   - LASSO / Elastic-Net regression: predict tuning parameters from gene expression profiles.
   - Identify top predictive genes via non-zero coefficients.
   - Cross-validate (leave-one-mouse-out) to test generalizability.
3. **Gene module analysis:**
   - Compute gene-gene correlation matrix across cells; identify co-expression modules (WGCNA or NMF).
   - Test if gene modules associate with specific functional properties more than individual genes.
4. **Visualization:**
   - Volcano plots of gene-tuning correlations (effect size vs. -log10 p-value).
   - Scatter plots for top gene-tuning relationships.
   - Heatmap: genes × tuning parameters, clustered on both axes.
   - Gene module–functional property association matrix.

---

#### A3. How does transcriptomic identity shape population coding geometry?

**Hypothesis A3:** Transcriptomically defined subclasses occupy distinct subspaces in population activity space, and the geometry of stimulus representations (e.g., orientation manifold) differs between excitatory subtypes and inhibitory subtypes.

**Analysis Plan:**
1. **Population activity vectors:**
   - Construct trial-averaged population response vectors (cells × stimuli matrix).
   - Apply PCA and UMAP to visualize population activity structure, color-coded by stimulus and by cell type.
2. **Representational similarity analysis (RSA):**
   - Compute representational dissimilarity matrices (RDMs) separately for each subclass.
   - Compare geometry of stimulus representations across subclasses using Procrustes analysis or centered kernel alignment (CKA).
3. **Decoding analysis:**
   - Train linear (LDA) and nonlinear (SVM-RBF) decoders on stimulus identity using only cells from one subclass. Compare decoding accuracy across subclasses.
   - Test: does adding inhibitory cells to excitatory populations improve decoding? Which subclass helps most?
4. **Visualization:**
   - Low-dimensional embedding of population activity (PCA/UMAP), split by subclass.
   - RDM comparison grids.
   - Decoding accuracy bar charts by subclass; decoding curves as function of population size.

---

#### A4. Do cell types differ in their temporal response dynamics? *(10 Hz data)*

**Hypothesis A4:** Different transcriptomic subclasses exhibit distinct temporal response profiles—inhibitory neurons (Pvalb, Sst) have shorter onset latencies and more transient responses than excitatory neurons (L2/3 IT, L4/5 IT). Within-subclass temporal diversity maps onto finer taxonomic levels and gene expression patterns.

**Analysis Plan:**
1. **PSTHs by subclass:** Using 10 Hz ΔF/F traces, compute trial-averaged peri-stimulus time histograms (PSTHs) for each cell, aligned to stimulus onset (−1 s to +3 s).
2. **Temporal metrics:** Extract onset latency (first timepoint > baseline + 2 SD), peak time, peak amplitude, and transient/sustained index (TSI = [early − late] / [early + late]).
3. **Cross-subclass comparison:** Kruskal-Wallis tests on temporal metrics across subclasses.
4. **Temporal clustering:** Cluster cells by normalized PSTH shape (correlation distance + Ward linkage); compare cluster composition to transcriptomic labels.
5. **Within-subclass diversity:** Quantify temporal diversity as mean pairwise correlation distance within each subclass.

---

### DOMAIN B: Contextual Adaptation and History Dependence

**Broad Question B:** *How do neural responses adapt to changing stimulus contexts, and do different cell types show distinct adaptation profiles?*

---

#### B1. Do responses differ between the first and second presentation of the same block type (context-dependent adaptation)?

**Hypothesis B1:** Responses to identical stimuli (matched contrast, TF, direction) differ between Block 0 vs. Block 2 (contrast-context blocks) and Block 1 vs. Block 3 (speed-context blocks), reflecting adaptation to the statistical structure of the preceding context. Inhibitory neurons (especially Sst and Vip) show stronger or faster contextual modulation than excitatory neurons.

**Analysis Plan:**
1. **Paired comparison (Block 0 vs. 2, Block 1 vs. 3):**
   - For each cell, compute mean response to each unique stimulus condition in each block.
   - Paired t-test / Wilcoxon signed-rank for each stimulus condition across block repetitions.
   - Compute an "adaptation index" = (Response_Block0 – Response_Block2) / (Response_Block0 + Response_Block2).
2. **Time course of adaptation within blocks:**
   - Bin trials by position within a block (early, middle, late).
   - Track response magnitude over trial position; fit exponential decay.
3. **Cell-type specificity:**
   - Compare adaptation indices across subclasses (ANOVA, post-hoc tests).
   - Within inhibitory types: are Vip cells more context-modulated than Pvalb or Sst?
4. **Cross-day consistency:**
   - Repeat across 3 days; test if adaptation is stable or changes with experience.
5. **Visualization:**
   - Paired stimulus response scatter plots (Block 0 vs. 2), colored by cell type.
   - Adaptation index distributions per subclass.
   - Time course plots (response vs. trial position) per cell type.
   - Heatmap: cells × trials, sorted by cell type and adaptation magnitude.

---

#### B2. Does the context (contrast-varying vs. speed-varying) alter tuning curves?

**Hypothesis B2:** In the contrast-context blocks, contrast response functions are sharper (lower C50) compared to responses at matched contrasts appearing in the speed-context blocks (where contrast is held at 0.8), reflecting contrast gain adaptation. Similarly, TF tuning shifts depending on context.

**Analysis Plan:**
1. **Context-dependent tuning curves:**
   - Compare CRF parameters (C50, Rmax, slope) from contrast-context blocks vs. matched single contrast (0.8) from speed-context blocks.
   - Compare TF tuning from speed-context blocks vs. matched single TF (1 Hz) from contrast-context blocks.
2. **Gain control modeling:**
   - Fit normalization models (Carandini & Heeger, 2012) to the context-dependent contrast responses.
   - Extract normalization pool parameters per cell type.
3. **Visualization:**
   - Overlay CRFs from contrast-context vs. single-point from speed-context per cell type.
   - Normalization model fits.
   - Scatter comparing C50 or preferred-TF across context conditions.

---

#### B3. Do responses change across days (multi-day adaptation / representational drift)?

**Hypothesis B3:** Trial-averaged responses to identical stimuli show systematic drift across the 3 recording days, with the degree of drift varying by cell type—excitatory L2/3 cells show more drift than L4/5 cells, and inhibitory cells are more stable.

**Analysis Plan:**
1. **Cross-session tracking:**
   - Use `cell_id - session_1/2/3` to match cells across days.
   - Compute trial-averaged responses per stimulus condition per day.
2. **Drift metrics:**
   - Pearson correlation of response vectors across days (day1 vs. day2, day1 vs. day3, day2 vs. day3).
   - Euclidean distance of tuning curves across days.
   - Signal correlation stability.
3. **Cell-type differences in drift:**
   - Compare stability metrics across subclasses.
4. **Population-level drift:**
   - Track movement of stimulus representations in PCA space across days.
   - CKA between population RDMs across days.
5. **Visualization:**
   - Scatter plots of cell-level response Day 1 vs. Day 3.
   - Stability index distributions per subclass.
   - Population PCA trajectories across days.

---

#### B4. How does the temporal shape of responses change with within-block adaptation and block context? *(10 Hz data)*

**Hypothesis B4:** Within-block adaptation manifests not just as reduced response amplitude but as a shift from transient to sustained temporal profiles. The contrast-context and speed-context blocks produce differently shaped PSTHs, with the contrast-context blocks eliciting more transient responses.

**Analysis Plan:**
1. **Within-block PSTH evolution:** Compare PSTHs from early (first 20%) vs. late (last 20%) trials within each block. Quantify changes in TSI, peak amplitude, and temporal shape.
2. **Context-dependent temporal shape:** Compare TSI between contrast-context (blocks 0, 2) and speed-context (blocks 1, 3) blocks for each cell. Paired Wilcoxon tests per subclass.
3. **Cell-type specificity:** Determine whether adaptation-related temporal changes are uniform across subclasses or preferentially affect certain types.

---

### DOMAIN C: Behavioral State Modulation (Running)

**Broad Question C:** *How does locomotion (running) modulate visual responses, and is this modulation cell-type specific?*

---

#### C1. Does running differentially modulate responses across cell types?

**Hypothesis C1:** Running multiplicatively increases responses in excitatory neurons (gain modulation) and Vip interneurons, while Sst interneurons are suppressed during running; Pvalb neurons show mixed or minimal modulation. This is consistent with a disinhibitory circuit: running → Vip activation → Sst suppression → excitatory disinhibition.

**Analysis Plan:**
1. **Running-modulation index (RMI):**
   - Split trials into running vs. stationary (using `is_running` / `avg_running` threshold).
   - For each cell, compute RMI = (R_run – R_stat) / (R_run + R_stat) per stimulus condition, then average.
   - Alternative: fit linear model Response ~ running_speed + stimulus, extract running coefficient.
2. **Gain vs. additive modulation:**
   - For each cell, fit: R(stim, run) = a × R(stim) + b, where a = gain, b = offset.
   - Compare gain vs. offset across subclasses.
3. **Running modulation of tuning curves:**
   - Compute full orientation × running interaction: does running change preferred orientation, OSI, or only amplitude?
   - Is the multiplicative gain stimulus-feature-specific?
4. **Visualization:**
   - RMI distributions per subclass (violin/ridge plots).
   - Tuning curves overlaid for running vs. stationary, per subclass.
   - Scatter: stationary response vs. running response, per cell type (slope = gain).
   - Population response heatmaps (cells sorted by RMI, split by type).

---

#### C2. Is the same VIP subtype that mediates running modulation also involved in context adaptation?

**Hypothesis C2 (Integrative — your VIP hypothesis):** A specific transcriptomic subtype of VIP interneurons (identifiable at the supertype or cluster level) simultaneously mediates: (a) running-state gain modulation, (b) context-dependent adaptation between blocks, and (c) experience-dependent changes across days. If so, this subtype has specific molecular markers (gene expression signature) that distinguish it from other VIP subtypes.

**Analysis Plan:**
1. **Multidimensional VIP characterization:**
   - For each VIP cell, compute: (i) RMI, (ii) adaptation index (Block 0→2), (iii) cross-day drift metric.
   - Cluster VIP cells by these three functional measures (k-means, hierarchical clustering).
2. **Map to transcriptomic subtypes:**
   - Test if functional VIP clusters correspond to supertype or cluster labels.
   - Differential gene expression between VIP functional subtypes.
3. **Correlation among modulations:**
   - Within VIP cells: is RMI correlated with adaptation index? With cross-day drift?
   - Compare to correlation in other inhibitory types (Sst, Pvalb, Lamp5).
4. **Circuit-level analysis:**
   - Examine if VIP cells with high running modulation are spatially near Sst cells with high suppression during running (spatial interaction analysis).
5. **Visualization:**
   - 3D scatter of VIP cells (RMI × adaptation × drift), colored by supertype/cluster.
   - Gene expression heatmap for VIP subtypes, sorted by functional cluster.
   - Network diagram of VIP-Sst-excitatory spatial proximity.

---

#### C3. Does moment-to-moment running speed modulate neural activity at sub-second timescales? *(10 Hz data)*

**Hypothesis C3:** Neural ΔF/F tracks instantaneous running speed with a characteristic lag of 100–300 ms, and this coupling is strongest in Vip neurons and weakest in Pvalb neurons. Running onset events trigger rapid ΔF/F changes that differ in latency and sign across subclasses.

**Analysis Plan:**
1. **Within-trial cross-correlation:** Compute cross-correlation between running speed and ΔF/F at ±500 ms lags within the stimulus period (10 Hz resolution). Average across trials per cell.
2. **Peak lag and coupling strength:** Extract peak correlation lag and magnitude per cell; compare across subclasses.
3. **Time-resolved gain curves:** At each 500 ms time window within the trial, bin trials by running speed and plot mean ΔF/F as a function of speed. Test whether the speed–response relationship changes over the course of the trial.
4. **Running-onset-triggered average:** Identify trials where running speed crosses a threshold during the stimulus; align ΔF/F to this onset event and compute onset-triggered averages per subclass.

---

#### C4. Do transcriptomic principal components predict running-state modulation?

**Hypothesis C4:** The first principal component of gene expression (tPC1, computed separately for glutamatergic and GABAergic classes) captures a gradient of running modulation strength that is not fully explained by subclass identity. Within excitatory neurons, tPC1 gradient correlates with RMI; within inhibitory neurons, tPC1 separates running-enhanced from running-suppressed cells.

**Analysis Plan:**
1. **Load tPC data:** Read `Glut_tPC1–5` and `GABA_tPC1–5` (first 5 PCs of the gene expression matrix, computed separately per class) from zarr multimodal stores. Create unified `tPC1–5` columns.
2. **Spearman correlations:** Correlate each tPC with RMI at three levels: whole-population, within-class (Glut vs GABA), and within-subclass (FDR-corrected).
3. **Multivariate regression:** OLS models predicting RMI from subclass dummies alone vs. subclass + tPC1–5. F-test for incremental R² from tPCs, separately for Glut and GABA classes.
4. **Visualization:**
   - Heatmap of within-subclass tPC–RMI correlations.
   - tPC1 vs tPC2 scatter colored by RMI (separate panels for Glut and GABA).
   - tPC1 vs RMI scatter per subclass with regression lines.
   - Violin plots of tPC1 split by running-enhanced / neutral / running-suppressed groups.

---

### DOMAIN D: Spatial Organization and Micro-Architecture

**Broad Question D:** *How does the spatial arrangement of cell types and their relative distances relate to functional properties and circuit interactions?*

---

#### D1. Is there spatial clustering of functionally similar neurons beyond what cell-type identity predicts?

**Hypothesis D1:** Neurons with similar orientation preferences are spatially clustered (salt-and-pepper with local bias) in L2/3, but less so in L4/5. When controlling for cell type, residual functional similarity correlates with spatial proximity (within ~50–100 µm).

**Analysis Plan:**
1. **Spatial autocorrelation of tuning:**
   - Compute Moran's I for preferred orientation, OSI, DSI, preferred TF as a function of spatial distance.
   - Compare observed spatial autocorrelation to shuffled cell-type labels (is there clustering beyond type?).
2. **Pairwise signal correlation vs. distance:**
   - For all cell pairs, compute signal correlation (correlation of trial-averaged tuning vectors) and noise correlation (trial-by-trial residual correlation).
   - Plot signal/noise correlation vs. pairwise Euclidean distance (3D).
   - Stratify by: same vs. different subclass, same vs. different layer (z-plane).
3. **Layer-specific analysis:**
   - Separate by depth; test if L2/3 vs. L4/5 show different spatial organization of orientation preference.
4. **Visualization:**
   - Spatial maps (x, y scatter, colored by preferred orientation, one per depth).
   - Signal/noise correlation vs. distance plots per subclass pair.
   - Moran's I correlogram.

---

#### D2. Does cell-type composition vary across the columnar extent and relate to functional properties?

**Hypothesis D2:** The ratio of inhibitory to excitatory neurons, and the specific inhibitory subclass distribution, varies systematically with cortical depth. Microcolumnar neighborhoods (~50 µm radius) with higher Vip density show stronger running modulation and context adaptation in local excitatory cells.

**Analysis Plan:**
1. **Depth profile of cell types:**
   - Histogram of cell counts per subclass across z-depth bins.
   - Chi-squared test for non-uniform distribution.
2. **Local neighborhood composition:**
   - For each excitatory cell, count inhibitory subtypes within radius R (test 30, 50, 100 µm).
   - Correlate local Vip/Sst/Pvalb density with excitatory cell's RMI, adaptation index, response magnitude.
3. **Visualization:**
   - Stacked bar chart of subclass proportions by depth.
   - 3D scatter of all cells, colored by subclass.
   - Scatter: local Vip density vs. excitatory cell RMI.

---

### DOMAIN E: Functional Connectivity and Circuit Interactions

**Broad Question E:** *What are the functional connectivity patterns between neurons, and how do these relate to cell-type identity, spatial proximity, and transcriptomic profiles?*

---

#### E1. Do noise correlations reveal cell-type-specific connectivity motifs?

**Hypothesis E1:** Noise correlations are strongest between neurons of the same subclass (especially within Pvalb and within excitatory subtypes), and weakest between Vip and Sst neurons (consistent with mutual inhibition). Noise correlation magnitude predicts spatial proximity and shared gene expression.

**Analysis Plan:**
1. **Noise correlation matrix:**
   - For each stimulus condition, compute trial-by-trial residuals (subtract trial-averaged response).
   - Compute pairwise Pearson correlation of residuals averaged across stimulus conditions.
2. **Cell-type-specific connectivity:**
   - Average noise correlations for each subclass pair (6×6 matrix for 6 subclasses).
   - Bootstrap confidence intervals; permutation test vs. shuffled labels.
3. **Relationship to spatial distance and gene expression:**
   - Partial correlation: noise correlation ~ distance + shared gene expression + cell-type match.
   - Mantel test between noise correlation matrix, distance matrix, gene-expression-similarity matrix.
4. **Visualization:**
   - Average noise correlation matrix (subclass × subclass), heatmap.
   - Noise correlation vs. Euclidean distance, stratified by cell-type pair.
   - Network graph of strongest pairwise connections, colored by cell type.

---

#### E2. Can cross-correlations and Granger causality at 10 Hz reveal directed functional interactions?

**Hypothesis E2:** Directed functional connectivity (inferred via Granger causality and sub-second cross-correlations on 10 Hz ΔF/F traces) reveals asymmetric interactions: L4/5 → L2/3 feedforward drive, and Sst → excitatory inhibitory influence. E–E pairs show narrow symmetric cross-correlograms, while E–I pairs are asymmetric with I cells lagging E cells.

**Analysis Plan:**
1. **Sub-second cross-correlations:**
   - Using the 10 Hz ΔF/F from zarr stores, concatenate stimulus-period traces across grating trials to form pseudo-continuous time series.
   - Compute pairwise cross-correlograms (CCGs) at ±500 ms lags.
   - Stratify by pair type (E–E, E–I, I–I) and examine CCG shape, peak lag, and asymmetry.
2. **Granger causality on 10 Hz traces:**
   - Apply pairwise Granger causality (max lag = 3 samples = 300 ms) on the concatenated 10 Hz data.
   - Identify significant directed edges (F-test, p < 0.01).
   - Aggregate directed edges by subclass pair (from → to).
3. **Cell-type-specific directed motifs:**
   - Compare directed connection probabilities to known circuit motifs (e.g., L4→L2/3, Vip→Sst, Sst→Pyr, Pvalb→Pyr).
   - Compute asymmetry index for each subclass pair.
4. **Spatial dependence:**
   - Plot GC connection probability vs. pairwise distance.
5. **Visualization:**
   - CCGs by pair category; peak lag distributions.
   - Directed connectivity matrix (subclass × subclass), asymmetry index.
   - GC connection probability vs. distance.

---

#### E3. How does population coupling relate to cell type and spatial position?

**Hypothesis E3:** Population coupling (correlation of single-cell activity with population mean) varies systematically: Pvalb neurons are most coupled (broad inhibition), Sst and Vip are less coupled, and among excitatory neurons, coupling decreases from L4/5 to L2/3. Highly coupled neurons are more centrally located in the field of view and express higher levels of synaptic markers.

**Analysis Plan:**
1. **Population coupling index:**
   - For each cell, compute correlation of its ΔF/F with the mean ΔF/F of all other cells (exclude self).
   - Compute coupling separately for running vs. stationary trials.
2. **Cell-type and spatial dependence:**
   - Compare coupling across subclasses.
   - Regress coupling on (x, y, z) position, cell type, and gene expression.
3. **Visualization:**
   - Population coupling distributions per subclass.
   - Spatial map colored by coupling strength.

---

#### E4. Does spontaneous activity structure during grey-screen epochs reveal cell-type-specific network dynamics?

**Hypothesis E4:** During extended spontaneous activity (360 s grey-screen epochs), population activity organizes into recurring assemblies whose composition reflects transcriptomic cell-type identity. Pvalb neurons participate broadly across assemblies while Sst neurons are selective, and assembly transitions occur at timescales that vary with running state.

**Analysis Plan:**
1. **Assembly detection:**
   - Apply PCA/ICA to the (3624 timepoints × n_cells) spontaneous ΔF/F matrix to extract co-activation patterns.
   - Use non-negative matrix factorization (NMF) to identify ~5–15 neural assemblies.
   - Compute assembly activation timecourses.
2. **Cell-type composition of assemblies:**
   - For each assembly, compute subclass participation weights; test for non-random composition (chi-squared vs. null).
   - Compare participation breadth (number of assemblies a cell belongs to) across subclasses.
3. **Running-state modulation of assemblies:**
   - Correlate assembly activation timecourses with concurrent running speed.
   - Identify running-associated vs. quiescence-associated assemblies.
4. **Cross-session stability:**
   - Compare assembly structure across 3 sessions (CKA of assembly weight matrices).
5. **Visualization:**
   - Raster plot of spontaneous activity sorted by assembly membership.
   - Assembly spatial maps and cell-type composition pie charts.
   - Assembly activation timecourses overlaid with running speed.

---

#### E5. Do catch-trial responses reveal cell-type-specific expectation signals?

**Hypothesis E5:** During catch trials (unexpected blank stimuli interleaved among gratings), specific cell types show systematic response deviations from baseline — particularly Sst neurons show suppression (release from stimulus-driven inhibition) and Vip neurons show transient activation (expectation/prediction signals). These catch-trial responses correlate with stimulus context (which block type the catch occurred in).

**Analysis Plan:**
1. **Catch-trial response profiles:**
   - Compute mean ΔF/F during catch trials (27 trials, 51 timepoints from −1 to +4 s) per cell. Compare to pre-stimulus baseline (−1 to 0 s).
   - Identify cells with significant catch-trial responses (paired t-test, stimulus period vs. baseline).
2. **Cell-type specificity:**
   - Compare catch-trial response magnitude, latency, and duration across subclasses.
   - Test if Sst cells show post-stimulus suppression and Vip cells show transient activation.
3. **Context dependence:**
   - Split catch trials by which block context they occurred in (contrast-context vs. speed-context).
   - Test if catch-trial responses differ by preceding context.
4. **Correlation with visual responsiveness:**
   - Correlate catch-trial response with each cell's GLM visual score or overall response amplitude.
5. **Visualization:**
   - PSTHs by subclass for catch trials.
   - Catch-trial response index distributions per subclass.
   - Scatter: grating response amplitude vs. catch-trial deviation.

---

### DOMAIN F: Computational Modeling (RNN-Based)

**Broad Question F:** *Can recurrent neural network models trained on this dataset reveal circuit mechanisms and generate testable predictions about V1 computation?*

---

#### F1. Can a task-trained RNN reproduce cell-type-specific tuning and connectivity?

**Hypothesis F1:** An RNN constrained with Dale's law (separate E/I units) and trained to perform stimulus classification reproduces the cell-type-specific tuning curves, noise correlations, and context-dependent adaptation observed in the data—but only when initialized with the observed E/I ratio and connectivity structure.

**Analysis Plan:**
1. **RNN architecture:**
   - Continuous-time RNN with ~1,000 units, split into E and I (matching observed ratios: ~90% E, ~4% Pvalb-like, ~3% Vip-like, ~2% Sst-like, ~1% Lamp5-like).
   - Enforce Dale's law (sign-constrained weights).
   - Input: stimulus features (orientation, contrast, TF) + running state.
   - Output: decoded stimulus identity (orientation classification).
2. **Training:**
   - Train on stimulus discrimination using the actual stimulus sequences from the data.
   - Use Adam optimizer with L2 regularization on weights.
   - Optionally add firing-rate penalty to keep dynamics biologically plausible.
3. **Analysis of trained network:**
   - Extract unit tuning curves; compare to data.
   - Compute noise correlations among RNN units; compare to observed.
   - Ablate specific unit types (Vip, Sst) and test impact on context adaptation.
4. **Predictions:**
   - The model predicts which cell types are necessary for context adaptation.
   - Can be tested against data by looking at context adaptation in animals/conditions with low VIP counts.
5. **Visualization:**
   - Tuning curves: real data vs. RNN units, split by type.
   - Weight matrix visualization (E→E, E→I, I→E, I→I subblocks).
   - Ablation impact bar charts.

---

#### F2. Can an RNN trained to predict neural activity learn biologically meaningful representations?

**Hypothesis F2:** An RNN trained to predict the population's ΔF/F activity from stimulus and running inputs develops internal representations whose geometry matches the observed population coding geometry, and whose recurrent connectivity structure mirrors the inferred functional connectivity from the data.

**Analysis Plan:**
1. **Data-driven RNN:**
   - Train an RNN to predict the population ΔF/F activity (all cells) given stimulus input and running speed.
   - Use teacher-forcing during training, then test on held-out trials.
2. **Representational comparison:**
   - Compare internal representations (hidden states) of the RNN to data via CKA and RSA.
   - Does the RNN develop cell-type-like clustering among its units?
3. **Connectivity analysis:**
   - Analyze learned recurrent weights; do motifs match E1/E2 results?
4. **Visualization:**
   - PCA of hidden states vs. PCA of real population.
   - Learned weight structure vs. noise correlation structure from data.

---

#### F3. Can an RNN predict full temporal trajectories of population ΔF/F? *(10 Hz data)*

**Hypothesis F3:** A 2-layer GRU-RNN trained on the 10 Hz trial-resolved ΔF/F (41 timepoints per trial, from −1 s to +3 s) learns to reproduce cell-type-specific temporal dynamics—including transient vs. sustained profiles, onset latencies, and running-speed-dependent modulation. Time-resolved CKA between RNN hidden states and real population activity peaks during the stimulus period and differs across time windows.

**Analysis Plan:**
1. **Temporal trajectory RNN:**
   - Input: (41, 6) per trial — cos(ori), sin(ori), contrast, log(TF), running speed(t), time(t).
   - Output: (41, n_cells) — predicted ΔF/F at each 100 ms timepoint.
   - Architecture: 2-layer GRU (256 hidden units) + 2-layer readout MLP.
   - Train on 80% of grating trials, test on held-out 20%.
2. **Temporal dynamics comparison:**
   - Compare predicted vs. real mean PSTHs per subclass.
   - Compute per-cell temporal Pearson r (across all time × trial pairs).
3. **Time-resolved CKA:**
   - At each of the 41 timepoints, compute CKA between real population activity and RNN hidden states across test trials.
   - Identify when the model best captures the neural code (stimulus onset? sustained period?).
4. **Visualization:**
   - Real vs. predicted PSTHs by subclass.
   - CKA as a function of time within trial.
   - Per-cell prediction quality by subclass.

---

### DOMAIN G: GLM-Based Encoding Analysis

**Broad Question G:** *What do pre-fitted encoding models reveal about the relative contributions of visual stimuli, behavioral state, and their interactions to neural activity across cell types?*

---

#### G1. Do cell types differ in the relative balance of visual vs. state-driven activity?

**Hypothesis G1:** Excitatory neurons (L2/3 IT, L4/5 IT) are primarily driven by visual stimuli (high `score_visual`, low `score_state`), while Vip neurons are disproportionately state-driven (high `score_state` relative to `score_visual`). Pvalb neurons show both high visual and state scores, consistent with their role integrating feedforward and modulatory inputs.

**Analysis Plan:**
1. **Visual vs. state R² decomposition:**
   - Load GLM `score_visual` and `score_state` per cell per session from zarr stores.
   - Compute visual/state ratio = `score_visual / score_total`, state fraction = `score_state / score_total`.
   - Compare across subclasses (Kruskal-Wallis, post-hoc Dunn's test).
2. **Cross-session stability of GLM scores:**
   - Correlate `score_visual` across sessions 1–3 per cell. Test if visual encoding is stable but state encoding drifts.
3. **Relationship to tuning metrics:**
   - Correlate GLM `score_visual` with OSI, max response amplitude, and reliability.
   - Correlate GLM `score_state` with RMI from Domain C1.
4. **Visualization:**
   - Scatter: `score_visual` vs. `score_state`, colored by subclass, per session.
   - Violin plots of visual fraction and state fraction by subclass.
   - Session-to-session correlation scatter for visual and state scores.

---

#### G2. What do condition-specific GLM kernels reveal about temporal encoding across cell types?

**Hypothesis G2:** GLM temporal kernels (30 basis functions per stimulus condition) reveal that L4/5 cells have sharper, more transient kernels than L2/3 cells, and that high-contrast conditions evoke faster kernel peaks than low-contrast conditions. Kernel shape systematically varies with temporal frequency in a cell-type-specific manner.

**Analysis Plan:**
1. **Extract and characterize per-condition kernels:**
   - For each cell, load the 30-dimensional coefficient vector for each stimulus condition.
   - Reconstruct temporal response kernels; extract peak time, peak amplitude, width, and transient/sustained ratio.
2. **Contrast-dependent kernel dynamics:**
   - Compare kernel peak times and shapes across the 5 contrast levels within contrast-context blocks.
   - Test if kernel sharpening with contrast differs by subclass.
3. **TF-dependent kernel dynamics:**
   - Compare kernel shapes across 5 TF levels within speed-context blocks.
   - Test if kernel temporal frequency matches stimulus TF (resonance).
4. **Visualization:**
   - Kernel heatmaps (conditions × time) per cell type.
   - Kernel peak time vs. contrast/TF curves by subclass.
   - PCA of kernel shapes, colored by subclass.

---

#### G3. Can GLM residuals (y − y_hat) reveal unmodeled computation?

**Hypothesis G3:** GLM residuals (activity not explained by visual + state models) contain structure that reflects nonlinear stimulus interactions and inter-neuronal coupling. Residual correlations between cell pairs are stronger for same-type pairs and spatially proximal neurons, suggesting the GLM residuals capture circuit-level interactions beyond stimulus-driven and state-driven components.

**Analysis Plan:**
1. **Residual computation:**
   - Compute `residual = y − y_hat` for each cell across the continuous recording.
   - Compare residual variance across subclasses (do some types have more unexplained activity?).
2. **Residual noise correlations:**
   - Compute pairwise correlations of GLM residuals between all cell pairs.
   - Compare to raw noise correlations (Domain E1): are residual correlations still cell-type-structured?
3. **Spatial structure of residuals:**
   - Correlate residual pairwise correlation with distance; compare to raw noise correlation–distance relationship.
4. **Residual dynamics:**
   - Segment residuals by block and examine whether unexplained variance increases with adaptation.
5. **Visualization:**
   - Subclass × subclass residual correlation matrix.
   - Scatter: GLM residual correlation vs. raw noise correlation per pair.
   - Residual variance by subclass, bar chart.

---

### DOMAIN H: Cell Morphology and Structure–Function Relationships

**Broad Question H:** *How do cell-body morphological features relate to transcriptomic identity, functional properties, and spatial position?*

---

#### H1. Does cell-body morphology predict transcriptomic cell type?

**Hypothesis H1:** Cell-body morphological features (volume, elongation, orientation) differ systematically across transcriptomic subclasses — particularly, Pvalb neurons are larger (more voxels) and rounder (lower PC1/PC2 ratio) than excitatory n eurons, while Sst neurons show more elongated somas. Morphological features can predict subclass identity above chance.

**Analysis Plan:**
1. **Morphological feature comparison:**
   - Load `n_voxels`, `size_pc1_um`, `size_pc2_um`, `size_pc3_um`, and `angle_deg_xy` from zarr morphology data.
   - Compute derived features: elongation index = `size_pc1_um / size_pc2_um`, volume proxy = `n_voxels`, soma flatness = `size_pc2_um / size_pc3_um`.
   - Compare across subclasses (Kruskal-Wallis, post-hoc tests).
2. **Morphology-based classification:**
   - Train Random Forest on morphological features to predict subclass; cross-validate (leave-one-mouse-out).
   - Compare accuracy to transcriptomic-based classification.
3. **Morphology and depth:**
   - Test if morphological features vary with cortical depth (`centroid_z_um`), controlling for cell type.
4. **Visualization:**
   - Violin plots of n_voxels, elongation, angle by subclass.
   - 2D scatter: elongation vs. volume, colored by subclass.
   - Confusion matrix from morphological classification.

---

#### H2. Does soma morphology correlate with functional properties independently of cell type?

**Hypothesis H2:** Within excitatory subclasses, larger cell bodies (more voxels) correspond to higher response amplitudes and stronger visual drive (GLM score_visual), while soma orientation angle in the XY plane correlates with preferred stimulus orientation — reflecting a potential link between soma geometry and dendritic arbor orientation.

**Analysis Plan:**
1. **Within-subclass morphology-function correlation:**
   - Within each subclass, correlate n_voxels with max ΔF/F response, OSI, GLM score_visual.
   - Within each subclass, correlate soma angle with preferred orientation (circular correlation).
2. **Multivariate model:**
   - Predict tuning metrics from morphology + gene expression + cell type; assess incremental R² from morphology.
3. **Visualization:**
   - Scatter: n_voxels vs. max response, per subclass.
   - Circular scatter: soma angle vs. preferred orientation.

---

## 3. Summary Table: Questions → Hypotheses → Analysis Pipeline

| # | Domain | Question (Short) | Key Hypothesis | Primary Methods | Visualization |
|---|---|---|---|---|---|
| A1 | Transcriptomic–Function | Cell-type-specific tuning | Subclass predicts tuning; finer taxonomy adds variance | von Mises / Naka-Rushton fits, ANOVA, variance decomposition | Tuning curves by type, violin plots, confusion matrices |
| A2 | Transcriptomic–Function | Gene expression predicts tuning | Specific genes (channels, neuropeptides) predict quantitative tuning | Spearman correlations (FDR), LASSO, gene modules | Volcano plots, heatmaps, scatter plots |
| A3 | Transcriptomic–Function | Population coding geometry by type | Subclasses occupy distinct activity subspaces | PCA/UMAP, RSA/CKA, decoding (LDA/SVM) | Embedding plots, RDMs, decoding curves |
| **A4** | **Transcriptomic–Function** | **Cell-type temporal dynamics** | **Inhibitory onset faster, more transient than excitatory** | **10 Hz PSTH, onset latency, TSI, temporal clustering** | **PSTHs by subclass, temporal cluster composition** |
| B1 | Context Adaptation | Block-to-block adaptation | Inhibitory cells (Vip, Sst) show stronger adaptation | Paired comparisons, adaptation index, trial dynamics | Paired scatter, adaptation distributions, time courses |
| B2 | Context Adaptation | Context alters tuning | Contrast context sharpens CRF; TF context shifts TF tuning | CRF/TF curve comparison, normalization model fits | Context-overlaid tuning curves, normalization fits |
| B3 | Context Adaptation | Multi-day drift | L2/3 drifts more than L4/5; inhibitory stable | Cross-day correlation, PCA tracking, CKA | Day-vs-day scatter, PCA trajectories |
| **B4** | **Context Adaptation** | **Within-trial temporal adaptation** | **Adaptation shifts PSTH from transient to sustained** | **10 Hz early vs late trial PSTH, TSI by block context** | **PSTH evolution, context-dependent TSI** |
| C1 | Running / State | Cell-type running modulation | Vip↑, Sst↓, excitatory gain modulation | RMI, gain/offset decomposition | Violin plots, tuning curves run vs. stat |
| C2 | Running / State (VIP) | Same VIP subtype for all? | One VIP cluster drives running + adaptation + drift | Multi-feature clustering, differential expression | 3D scatter, gene heatmaps |
| **C3** | **Running / State** | **Moment-to-moment running coupling** | **ΔF/F tracks running with 100–300 ms lag; Vip strongest** | **10 Hz cross-correlation, running-onset-triggered avg** | **CCGs by subclass, time-resolved gain curves** |
| C4 | Running / State | tPCs predict running modulation | tPC1 gradient correlates with RMI beyond subclass | Spearman, OLS (subclass + tPCs), F-test | Heatmap, tPC scatter colored by RMI, violins |
| D1 | Spatial Organization | Functional clustering in space | Salt-and-pepper with local bias, beyond cell type | Moran's I, signal/noise corr vs. distance | Spatial maps, correlograms |
| D2 | Spatial Organization | Local composition affects function | High local Vip density → stronger modulation | Neighborhood composition, correlation with RMI | Depth profiles, scatter density plots |
| E1 | Connectivity | Noise correlation motifs | Same-type strongest; Vip-Sst weakest | Noise correlations, Mantel test, partial correlation | Subclass×subclass heatmaps, distance plots |
| E2 | Connectivity | Directed connectivity (10 Hz) | L4/5→L2/3 feedforward; E–I asymmetric CCGs | 10 Hz cross-correlations, Granger causality | CCGs by pair type, directed matrix, distance plots |
| E3 | Connectivity | Population coupling | Pvalb most coupled; L2/3 less than L4/5 | Coupling index, regression | Coupling distributions, spatial maps |
| F1 | RNN Modeling | Task-trained RNN | Dale's law RNN reproduces cell-type tuning/connectivity | RNN training, ablation, tuning comparison | Tuning comparison, weight matrices, ablation charts |
| F2 | RNN Modeling | Data-driven RNN | RNN internal representations match data geometry | Prediction RNN, CKA/RSA, weight analysis | PCA comparison, weight structure |
| **F3** | **RNN Modeling** | **Temporal trajectory RNN** | **RNN learns transient/sustained profiles matching data** | **10 Hz GRU training, time-resolved CKA** | **Predicted vs real PSTHs, CKA over time** |
| G1 | GLM Encoding | Visual vs. state drive by type | Excitatory → visual-driven; Vip → state-driven | GLM score decomposition, cross-session stability | Visual vs state scatter, violin plots |
| G2 | GLM Encoding | Condition-specific temporal kernels | L4/5 kernels sharper than L2/3; contrast sharpens peaks | Kernel shape extraction, peak time comparison | Kernel heatmaps, peak time curves |
| G3 | GLM Encoding | GLM residual structure | Residuals capture circuit interactions beyond stimulus/state | Residual correlations, spatial analysis | Residual correlation matrices, variance by type |
| H1 | Morphology | Soma morphology predicts cell type | Pvalb larger/rounder; Sst more elongated | Morphological feature stats, Random Forest classification | Violin plots, confusion matrices |
| H2 | Morphology | Morphology correlates with function | Larger somas → stronger visual drive; angle → pref. ori | Within-subclass correlations, multivariate regression | Scatter plots, circular correlations |
| E4 | Connectivity | Spontaneous activity assemblies | Assemblies reflect cell-type identity; Pvalb broadly coupled | PCA/ICA/NMF on grey-screen data, assembly detection | Rasters, assembly spatial maps, timecourses |
| E5 | Connectivity | Catch-trial expectation signals | Sst suppressed, Vip activated on catch; context-dependent | Catch-trial PSTH analysis, response classification | PSTHs, catch response distributions |

---

## 4. Recommended Analysis Priority and Workflow

### Phase 1: Foundation (Essential, do first)
1. **A1** — Cell-type tuning characterization (establishes baseline functional description)
2. **C1** — Running modulation by cell type (leverages trial-level running data readily available)
3. **E1** — Noise correlations (foundational connectivity measure)
4. **G1** — GLM visual vs. state decomposition (leverages pre-fitted models in zarr; fast to execute)

### Phase 2: Core Science (Main findings)
5. **B1** — Context adaptation (leverages the unique block structure)
6. **D1** — Spatial functional organization
7. **A2** — Gene-tuning relationships
8. **C2** — VIP integration hypothesis (your central hypothesis, needs A1+B1+C1 results)
9. **C4** — Transcriptomic PCs and running modulation (requires C1 + zarr tPC data)
10. **H1** — Soma morphology and cell-type identity (fast analysis from zarr morphology data)

### Phase 3: Deep Mechanistic
11. **B2** — Normalization / gain control modeling
12. **E2** — Directed connectivity
13. **A3** — Population geometry by type
14. **D2** — Local neighborhood effects
15. **E3** — Population coupling
16. **G2** — GLM condition-specific temporal kernels
17. **G3** — GLM residual structure analysis
18. **E4** — Spontaneous activity assemblies (grey-screen data)
19. **E5** — Catch-trial expectation signals
20. **H2** — Morphology–function within-type correlations

### Phase 4: Computational Modeling
21. **F1** — Task-trained RNN
22. **F2** — Data-driven RNN
23. **F3** — Temporal trajectory RNN (10 Hz)
24. **B3** — Multi-day drift (requires 4th mouse or natural movie data)

---

## 5. Key Software and Tools

| Tool | Purpose |
|---|---|
| **Python / Pandas / NumPy** | Data handling, gene expression analysis |
| **zarr** | Reading multimodal zarr data stores (morphology, GLM, 10 Hz physiology) |
| **NumPy / SciPy / scikit-learn** | Statistics, curve fitting, decoding, clustering |
| **Matplotlib / Seaborn / Plotly** | Visualization |
| **statsmodels** | Granger causality, regression, ANOVA |
| **PyTorch / JAX** | RNN modeling |
| **PySAL / libpysal** | Spatial autocorrelation (Moran's I) |
| **WGCNA (rpy2) or NMF** | Gene co-expression modules |

---

*Document generated based on inspection of 3 zarr multimodal data stores (778174, 786297, 797371) containing 2,824 neurons, 2,186 grating trials + 27 catch trials + ~360 s spontaneous activity per session, 299 Xenium genes, 3D cell coordinates, cell-body morphology, and pre-fitted GLM encoding models.*
