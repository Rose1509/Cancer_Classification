(() => {
    const configEl = document.getElementById("app-config");
    const config = configEl ? JSON.parse(configEl.textContent) : {};
    window.DEFAULT_MODEL = config.default_model || "improved";
    window.LOADED_MODELS = config.loaded_models || [];

    const form = document.getElementById("upload-form");
    if (!form) return;
    const fileInput = document.getElementById("image-input");
    const browseBtn = document.getElementById("browse-btn");
    const dropzone = document.getElementById("dropzone");
    const imageWrapper = document.getElementById("image-wrapper");
    const previewImg = document.getElementById("preview-image");
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const analyzingOverlay = document.getElementById("analyzing-overlay");
    const analyzingProgressBar = document.getElementById("analyzing-progress-bar");
    const scanId = document.getElementById("scan-id");
    const malignantValue = document.getElementById("malignant-value");
    const malignantBar = document.getElementById("malignant-bar");
    const confidenceValue = document.getElementById("confidence-value");
    const predictionLabel = document.getElementById("prediction-label");
    const recommendation = document.getElementById("recommendation-text");
    const errorBox = document.getElementById("error-box");
    const analyzeBtn = document.getElementById("analyze-btn");
    const modelCards = document.querySelectorAll(".model-card[data-model]");
    const resultsPanel = document.querySelector(".prediction-results-container");

    if (!imageWrapper || !previewImg || !previewPlaceholder) {
        console.error("Preview elements missing. Hard-refresh the page (Ctrl+F5).");
        return;
    }

    const ANALYZE_DURATION_MS = 5500;

    // Start with no model selected; require explicit user selection
    let selectedModel = null;
    let previewObjectUrl = null;
    let progressInterval = null;

    function showError(message) {
        errorBox.textContent = message;
        errorBox.hidden = false;
    }

    function clearError() {
        errorBox.hidden = true;
        errorBox.textContent = "";
    }

    function delay(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function setAnalyzingUI(active) {
        analyzeBtn.disabled = active;
        analyzeBtn.textContent = active ? "Analyzing..." : "Analyze Image";
        resultsPanel.classList.toggle("is-analyzing", active);

        if (active) {
            predictionLabel.textContent = "Analyzing";
            predictionLabel.className = "prediction-badge analyzing";
            analyzingOverlay.hidden = false;
            imageWrapper.classList.add("is-analyzing");
            startProgressAnimation();
            return;
        }

        analyzingOverlay.hidden = true;
        imageWrapper.classList.remove("is-analyzing");
        stopProgressAnimation();
    }

    function startProgressAnimation() {
        stopProgressAnimation();
        let progress = 0;
        analyzingProgressBar.style.width = "0%";
        const step = 100 / (ANALYZE_DURATION_MS / 50);
        progressInterval = setInterval(() => {
            progress = Math.min(progress + step, 100);
            analyzingProgressBar.style.width = `${progress}%`;
            if (progress >= 100) stopProgressAnimation();
        }, 50);
    }

    function stopProgressAnimation() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }

    function selectModel(card) {
        modelCards.forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        selectedModel = card.dataset.model;
    }

    modelCards.forEach((card) => {
        card.addEventListener("click", () => {
            if (!window.LOADED_MODELS.includes(card.dataset.model)) {
                showError(`Model "${card.dataset.model}" is not available. Train or add its weights file.`);
                return;
            }
            clearError();
            selectModel(card);
            console.debug('Model selected:', card.dataset.model);
        });
    });

    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", (e) => {
        if (e.target === browseBtn) return;
        fileInput.click();
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        if (e.dataTransfer.files?.length) {
            // Use DataTransfer to populate the file input for compatibility
            try {
                const dt = new DataTransfer();
                Array.from(e.dataTransfer.files).forEach((f) => dt.items.add(f));
                fileInput.files = dt.files;
            } catch (err) {
                // Fallback: some browsers may allow direct assignment
                try { fileInput.files = e.dataTransfer.files; } catch (e) { /* ignore */ }
            }
            previewFile(e.dataTransfer.files[0]);
            console.debug('Dropped file:', e.dataTransfer.files[0].name);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files?.length) previewFile(fileInput.files[0]);
    });

    function showPreview() {
        imageWrapper.classList.add("has-image");
        previewPlaceholder.style.display = "none";
        previewImg.style.display = "block";
    }

    function previewFile(file) {
        clearError();

        if (!file || !file.type.startsWith("image/")) {
            showError("Please upload a valid image file (PNG, JPG, etc.).");
            return;
        }

        scanId.textContent = `ID: ${file.name}`;

        if (previewObjectUrl) {
            URL.revokeObjectURL(previewObjectUrl);
            previewObjectUrl = null;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.onload = () => {
                try {
                    const w = previewImg.naturalWidth || previewImg.width;
                    const h = previewImg.naturalHeight || previewImg.height;
                    if (w && h) {
                        imageWrapper.style.aspectRatio = `${w} / ${h}`;
                        console.debug('Set wrapper aspect ratio', w, h);
                    }
                } catch (err) {
                    console.debug('Could not set aspect ratio', err);
                }
                showPreview();
                console.debug('Preview loaded for', file.name);
            };
        };
        reader.onerror = () => showError("Could not load image preview.");
        reader.readAsDataURL(file);
    }

    function updateResults(data) {
        const isMalignant = data.label === "malignant";
        const predictedName = data.label.charAt(0).toUpperCase() + data.label.slice(1);

        // Show the probability for the predicted class (confidence)
        const metricTitle = document.querySelector('.probability-section .metric-title');
        if (metricTitle) metricTitle.textContent = `${predictedName} Probability`;

        malignantValue.textContent = `${data.confidence}%`;
        malignantBar.style.width = `${data.confidence}%`;
        malignantBar.classList.toggle("malignant", isMalignant);
        malignantBar.classList.toggle("benign", !isMalignant);

        confidenceValue.textContent = `${data.confidence}%`;
        predictionLabel.textContent = predictedName;
        predictionLabel.className = `prediction-badge ${data.label}`;

        const modelName = data.model === "improved" ? "Improved CNN"
            : data.model === "resnet" ? "ResNet-18" : "Baseline Model";

        recommendation.innerHTML = isMalignant
            ? `<strong>Recommendation:</strong> ${modelName} detected patterns consistent with <strong>malignant</strong> tissue (${data.malignant_probability}% malignant probability). Clinical review advised.`
            : `<strong>Recommendation:</strong> ${modelName} classified this scan as <strong>benign</strong> (${data.confidence}% confidence). Routine follow-up per standard protocol.`;
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        clearError();

        if (!fileInput.files?.length) {
            showError("Please select an image to analyze.");
            return;
        }

        if (!selectedModel) {
            showError("Please select a model before analyzing the image.");
            return;
        }

        if (!imageWrapper.classList.contains("has-image")) {
            previewFile(fileInput.files[0]);
        }

        const formData = new FormData();
        formData.append("image", fileInput.files[0]);
        formData.append("model", selectedModel);

        setAnalyzingUI(true);

        try {
            const [response] = await Promise.all([
                fetch("/predict", { method: "POST", body: formData }),
                delay(ANALYZE_DURATION_MS),
            ]);

            const data = await response.json();
            if (!response.ok) {
                showError(data.error || "Prediction failed.");
                predictionLabel.textContent = "Pending";
                predictionLabel.className = "prediction-badge pending";
                return;
            }
            updateResults(data);
        } catch {
            showError("Could not reach the server. Make sure the app is running.");
            predictionLabel.textContent = "Pending";
            predictionLabel.className = "prediction-badge pending";
        } finally {
            setAnalyzingUI(false);
        }
    });
})();
