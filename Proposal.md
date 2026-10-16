# **Quantum vs Classical Approaches for Streaming Data Analytics: An Experimental Study** 

Team Members: 

1. Ashwin S - 2023BCS0044 

2. Adnan Omar Chozhimadam - 2023BCS0035 

3. Mohamed Sheik Mubaris - 2023BCS0062 

4. Hulke Sumit - 2023BCD0026 

## 1. Problem Statement 

The systems that deal with streaming data analytics need to classify, predict or detect anomalies in high-velocity data under time and space limitations and must do so while adapting in an incremental way due to incoming new data. The classical framework of streaming machine learning (e.g., incremental decision trees, online kernel methods) has many years of scientific and practical experience with such problems. Quantum machine learning (QML) is recently suggested as an alternative to streaming ML which possesses an advantage in sample complexity, expressiveness, and parameterization of the models especially in kernel-based and variational QML. However, the majority of published results of QML are done in offline and batch-style mode on NISQ simulation and it is not clear yet how much of this advantage survives in the regime of true streaming. 

This research involves the empirical evaluation of traditional stream baselines versus quantum as well as hybrid quantum-classical paradigms using real world streaming/sensor datasets. Performance aspects such as accuracy, efficiency (in terms of throughput and per-instance latency) and sample efficiency are being considered, and in addition to this, a novel hybrid modification involving the adaptation of a quantum kernel based approach into an incremental window-based training framework is being explored. 

## 2. Base Papers 

2.1 Domingos, P. & Hulten, G. (2000). "Mining High-Speed Data Streams." Proceedings of the 6th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '00), pp. 71–80. [Baseline to be reproduced] 

This paper introduces the Hoeffding Tree algorithm and the Very Fast Decision Tree (VFDT) system, which learn decision trees incrementally from a single pass over a data stream using constant time and memory per example. Splitting decisions at each node are made only once the Hoeffding bound guarantees, with high statistical confidence, that the locally best attribute is the same one a batch learner would choose given infinite data. The authors show theoretically and empirically that VFDT trees converge to trees nearly identical to their batch counterparts while processing streams orders of magnitude faster than existing batch or windowed incremental learners, making it one of the most cited and reproduced streaming classification methods and the natural classical baseline for this project. 

**a. Research question:** Can decision trees be induced incrementally, in a single pass and with bounded memory, from a high-speed data stream while guaranteeing accuracy asymptotically close to a batch learner trained on the same (effectively infinite) data? 

**b. Data used:** Synthetic data streams generated from randomly constructed "true" decision trees of varying size and noise level, used to measure agreement with the batch tree; a large real dataset (a web/clickstream log with on the order of a million examples) used to demonstrate scalability. 

**c. Metrics used:** Classification accuracy and tree-size agreement relative to the batch (offline) decision tree, memory consumption, and learning throughput (examples processed per second / total learning time). 

**d. Key findings:** The Hoeffding bound lets VFDT select splitting attributes with a provable, tunable confidence level using only a small 

number of examples per node; the resulting trees are near-identical in accuracy to batch trees; VFDT scales linearly in the number of examples with constant memory per node, allowing it to mine millions of examples in a fraction of the time required by batch learners such as C4.5, with negligible accuracy loss. 

**e. Model:** Hoeffding Tree (VFDT) an online, single-pass, statistically-bounded incremental decision tree. 

_Source: ACM Digital Library / KDD 2000 proceedings (winner, KDD 2015 "Test of Time" award)._ 

2.2 Herbst, S., De Maio, V. & Brandic, I. (2024). "Streaming IoT Data and the Quantum Edge: A Classic/Quantum Machine Learning Use Case." (arXiv:2402.15542; Springer LNCS chapter). 

This paper studies how variational quantum machine learning can be embedded into an Edge-computing pipeline that sits between streaming IoT sensor sources and quantum hardware, proposing the "Quantum Edge" as an intermediate layer that handles data encoding, hyperparameter tuning, and error mitigation before and after quantum execution. The authors run a systematic, 672-configuration hyperparameter search for a variational quantum regressor (feature map, entanglement pattern, ansatz, and classical optimizer) on the Seoul Bike Sharing Demand dataset, a public IoT-style sensor dataset, using Qiskit Aer simulators, and report which encoding and optimizer choices most strongly affect predictive accuracy and runtime. 

**a. Research question:** How can quantum machine learning be integrated into an edge-computing pipeline for streaming IoT data, and how do data-encoding choices (feature map, entanglement) and circuit/optimizer hyperparameters affect the predictive performance and runtime of the resulting quantum regression model? 

**b. Data used:** Seoul Bike Sharing Demand dataset (UCI Machine Learning Repository): 13 weather/temporal attributes, 8,760 hourly instances, used as a proxy for IoT sensor streams; dimensionality 

reduced to 7 features via PCA (to fit 7-qubit hardware), with 400 training and 250 evaluation points and the target normalized to [0,1]. 

**c. Metrics used:** Mean Squared Error (MSE) and Mean Absolute Error (MAE) of the regression target, and wall-clock training/runtime, compared across 672 hyperparameter configurations (feature map × entanglement × ansatz × optimizer). 

**d. Key findings:** The ZFeatureMap encoding significantly outperformed the ZZFeatureMap, though the gap depended on the optimizer used; the SPSA optimizer produced the best configurations overall, COBYLA was competitive and faster, while Nelder-Mead was clearly worse despite long runtimes; ansatz choice (EfficientSU2, RealAmplitudes, TwoLocal, PauliTwoDesign) mattered less than encoding/optimizer choice; no single configuration dominated, motivating automatic hyperparameter tuning at the Edge layer. 

**e. Model:** Variational Quantum Regression (VQA-based): a parametrized quantum circuit (ansatz) combined with a quantum feature map and a classical optimizer, executed on Qiskit Aer simulators. 

_Source: arXiv:2402.15542; published as a chapter in Springer LNCS proceedings (2024)._ 

2.3 Tscharke, K., Issel, S. & Debus, P. (2023). "Semisupervised Anomaly Detection using Support Vector Regression with Quantum Kernel." IEEE International Conference on Quantum Computing and Engineering (QCE 2023). (arXiv:2308.00583). 

This paper proposes Quantum Support Vector Regression (QSVR) for semisupervised anomaly detection: a support vector regressor with a quantum-estimated kernel is trained only on normal data, and anomaly scores are derived from its reconstruction loss. The authors benchmark QSVR against a quantum autoencoder (quantum baseline) and against an RBF-kernel SVR and a classical autoencoder (classical baselines) on eleven real-world anomaly-detection datasets, reporting AUC as the primary metric. This paper anchors the anomaly-detection / quantum-kernel side of our comparison, directly pitting a 

quantum-kernel classical-ML hybrid against purely classical kernel and deep-learning baselines. 

**a. Research question:** Can a quantum-kernel support vector regressor, trained in a semisupervised setting and scored by reconstruction loss, match or outperform classical kernel-based and autoencoder-based anomaly detectors, and how does it compare to a quantum autoencoder baseline? 

**b. Data used:** Ten real-world semisupervised anomaly-detection benchmark datasets plus one toy dataset (standard tabular AD benchmark suite). 

**c. Metrics used:** Area Under the ROC Curve (AUC) per dataset and mean AUC averaged across all eleven datasets. 

**d. Key findings:** QSVR achieved the highest mean AUC of all four compared models (QSVR, quantum autoencoder, RBF-kernel SVR, classical autoencoder) and outperformed the quantum autoencoder baseline on 9 of the 11 datasets, indicating that quantum-kernel estimation can provide a measurable, consistent advantage for semisupervised anomaly detection even under NISQ-scale simulation constraints. 

**e. Model:** Support Vector Regression with a quantum-estimated kernel (QSVR), benchmarked against an RBF-kernel SVR, a classical autoencoder, and a quantum autoencoder. 

_Source: arXiv:2308.00583 (v2); accepted at IEEE QCE 2023, DOI 10.1109/QCE57702.2023.00075._ 

## 3. Dataset / Stream Source 

To keep our reproduction of the Domingos & Hulten baseline directly comparable to a real streaming benchmark, and to reuse the exact dataset from one of our quantum base papers, we propose three complementary sources, each replayed example-by-example to simulate an online stream: 

- Seoul Bike Sharing Demand (UCI Machine Learning Repository) the same dataset used in the Herbst et al. quantum-regression paper; 8,760 hourly records of weather and calendar features used to predict bike-rental demand. Reused directly to reproduce and extend that paper's quantum-vs-classical regression comparison under a streaming protocol. 

- Electricity Market Dataset (Australian New South Wales electricity market, "ELEC2") a standard, widely used concept-drift streaming classification benchmark, used to reproduce the Hoeffding Tree / VFDT baseline in a genuinely non-stationary streaming setting. 

- A public anomaly-detection benchmark stream (e.g., NSL-KDD network intrusion data, or one of the eleven benchmark sets used by Tscharke et al., subject to availability)  replayed sequentially to evaluate the quantum-kernel anomaly detector against classical baselines under streaming conditions. 

All sources are public, free to redistribute for academic use, and small enough (after PCA-based dimensionality reduction where needed) to run on quantum simulators within course compute budgets. 

## 4. Metrics 

- Predictive quality: prequential (test-then-train) accuracy and F1-score for classification; MSE and MAE for regression; AUC for anomaly detection. 

- Computational efficiency: per-example latency and overall throughput (examples/second), separated into encoding time vs. model update/inference time for the quantum conditions. 

- Sample efficiency: performance as a function of the number of training examples seen, to test the QML literature's claim of faster convergence with fewer samples. 

- Resource footprint: number of trainable parameters / circuit depth and qubit count for quantum models, versus tree size / support-vector count for classical models. 

- Robustness to drift: performance before/after synthetic or real concept drift (relevant to the Electricity dataset and the Hoeffding Tree reproduction). 

### 5. Chosen Tool Combination 

|**Tool**|**Role in project**|**Classical / Quantum**|
|---|---|---|
|**River**|Streaming ML in Python: reproduces Hoeffding Tree /<br>VFDT, provides prequential evaluation and drift<br>detectors (ADWIN) for the classical baseline.|Classical|
|**scikit-learn**|Classical SVR/SVM, Random Forest, PCA for<br>dimensionality reduction, and general preprocessing<br>shared across classical and quantum pipelines.|Classical|
|**Qiskit + Qiskit Machine**<br>**Learning**|Implements the variational quantum regressor (feature<br>maps, ansätze, optimizers) to reproduce and extend the<br>Herbst et al. setup on Qiskit Aer simulators.|Quantum|
|**PennyLane**|Implements the quantum-kernel SVR (QSVR) with<br>autodiff support, used to reproduce/extend the Tscharke<br>et al. quantum-kernel anomaly detector and to build the<br>streaming/hybrid enhancement.|Quantum / Hybrid|



River and scikit-learn cover the classical streaming and batch baselines; Qiskit and PennyLane cover two complementary quantum machine learning styles (variational circuits and quantum kernels) so the comparison is not tied to a single QML framework or vendor. 

## 6. Expected Enhancement 

Both quantum base papers train their quantum models in a batch (offline) manner and only motivate their relevance via streaming/IoT use cases, without actually running them under an incremental, windowed training protocol. Our planned enhancement is to wrap the reproduced quantum-kernel (QSVR) and variational-regression models in a sliding-window, incremental training/scoring loop retraining or updating on fixed-size recent windows as new stream examples arrive, mirroring how the classical Hoeffding Tree baseline naturally operates and to explicitly measure the data-encoding overhead this introduces, which Herbst et al. identify as a key open bottleneck but do not quantify. This lets us report, for the first time in this comparison, whether any 

quantum-side predictive advantage (as found by Tscharke et al.) survives once realistic streaming latency and incremental-update constraints are imposed, and where the classical/quantum trade-off line actually falls for this class of problems. 

## 7. Scope Addendum (implemented scope)

The implemented project narrows and sharpens the scope above: **ELEC2 only**, **binary
classification**, and an **event-triggered (ADWIN) adaptation** mechanism instead of periodic
window retraining. The classical Hoeffding-tree baseline (Domingos & Hulten, 2000) is
reproduced; the Herbst et al. and Tscharke et al. quantum models are adapted to the streaming
ELEC2 classification setting. A classical RBF-kernel control pair was added to the model
matrix for the sample-efficiency comparison. The full list of deviations and their
justification is maintained in `PLAN.md`, Section 11 ("Deviations from Proposal & Scope
Notes"). 

