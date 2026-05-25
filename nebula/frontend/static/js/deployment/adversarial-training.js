// Adversarial Training Module
const AdversarialTrainingManager = (function() {
    const DEFAULT_ADVERSARIAL_TRAINING_CONFIG = {
        enabled: false,
        domain: "image",
        attack: "fgsm",
        epsilon: 0.03,
        alpha: null,
        steps: 1,
        mode: "mixed",
        clean_weight: 0.5,
        adversarial_weight: 0.5,
        apply_probability: 1.0,
        clip_min: 0.0,
        clip_max: 1.0,
        log_adversarial_metrics: true
    };

    const IMAGE_DATASETS = new Set(["MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100"]);

    function initializeAdversarialTraining() {
        setupAdversarialTrainingSwitch();
        setupAttackSelector();
        setupDatasetAwareness();
        setAdversarialTrainingConfig(DEFAULT_ADVERSARIAL_TRAINING_CONFIG);
    }

    function setupAdversarialTrainingSwitch() {
        const adversarialTrainingSwitch = document.getElementById("adversarialTrainingSwitch");
        if (!adversarialTrainingSwitch) return;

        adversarialTrainingSwitch.addEventListener("change", function() {
            if (this.checked && window.DpManager) {
                window.DpManager.setDpConfig({enabled: false});
            }
            toggleAdversarialTrainingSettings(this.checked);
        });
    }

    function setupAttackSelector() {
        const attackSelect = document.getElementById("adversarialTrainingAttack");
        if (!attackSelect) return;

        attackSelect.addEventListener("change", function() {
            toggleAttackSettings(this.value);
        });
    }

    function setupDatasetAwareness() {
        const datasetSelect = document.getElementById("datasetSelect");
        if (!datasetSelect) return;

        datasetSelect.addEventListener("change", updateDatasetAvailability);
        updateDatasetAvailability();
    }

    function toggleAdversarialTrainingSettings(enabled) {
        const settings = document.getElementById("adversarial-training-settings");
        if (!settings) return;

        settings.style.display = enabled ? "block" : "none";
        toggleAttackSettings(document.getElementById("adversarialTrainingAttack")?.value || "fgsm");
    }

    function toggleAttackSettings(attack) {
        const pgdSettings = document.getElementById("adversarial-training-pgd-settings");
        if (!pgdSettings) return;

        pgdSettings.style.display = attack === "pgd" ? "block" : "none";
    }

    function updateDatasetAvailability() {
        const dataset = document.getElementById("datasetSelect")?.value;
        const domain = IMAGE_DATASETS.has(dataset) ? "image" : "tabular";
        const adversarialTrainingSwitch = document.getElementById("adversarialTrainingSwitch");
        const datasetNote = document.getElementById("adversarial-training-dataset-note");
        const domainInput = document.getElementById("adversarialTrainingDomain");

        if (datasetNote) {
            datasetNote.style.display = "none";
        }
        if (domainInput) {
            domainInput.value = domain;
        }

        if (!adversarialTrainingSwitch) return;
        adversarialTrainingSwitch.disabled = false;
    }

    function numberValue(id, fallback) {
        const value = parseFloat(document.getElementById(id)?.value);
        return Number.isFinite(value) ? value : fallback;
    }

    function integerValue(id, fallback) {
        const value = parseInt(document.getElementById(id)?.value, 10);
        return Number.isFinite(value) ? value : fallback;
    }

    function optionalNumberValue(id, fallback) {
        const rawValue = document.getElementById(id)?.value;
        if (rawValue === undefined || rawValue === null || rawValue === "") {
            return fallback;
        }
        const value = parseFloat(rawValue);
        return Number.isFinite(value) ? value : fallback;
    }

    function getAdversarialTrainingConfig() {
        const config = {
            enabled: Boolean(document.getElementById("adversarialTrainingSwitch")?.checked),
            domain: document.getElementById("adversarialTrainingDomain")?.value || DEFAULT_ADVERSARIAL_TRAINING_CONFIG.domain,
            attack: document.getElementById("adversarialTrainingAttack")?.value || DEFAULT_ADVERSARIAL_TRAINING_CONFIG.attack,
            epsilon: numberValue("adversarialTrainingEpsilon", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.epsilon),
            alpha: optionalNumberValue("adversarialTrainingAlpha", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.alpha),
            steps: integerValue("adversarialTrainingSteps", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.steps),
            mode: document.getElementById("adversarialTrainingMode")?.value || DEFAULT_ADVERSARIAL_TRAINING_CONFIG.mode,
            clean_weight: numberValue("adversarialTrainingCleanWeight", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.clean_weight),
            adversarial_weight: numberValue("adversarialTrainingAdversarialWeight", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.adversarial_weight),
            apply_probability: numberValue("adversarialTrainingApplyProbability", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.apply_probability),
            clip_min: numberValue("adversarialTrainingClipMin", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.clip_min),
            clip_max: numberValue("adversarialTrainingClipMax", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.clip_max),
            log_adversarial_metrics: Boolean(document.getElementById("adversarialTrainingLogMetrics")?.checked)
        };

        if (config.alpha === null || config.attack !== "pgd") {
            delete config.alpha;
        }
        return config;
    }

    function setAdversarialTrainingConfig(config = DEFAULT_ADVERSARIAL_TRAINING_CONFIG) {
        const adversarialTrainingConfig = {
            ...DEFAULT_ADVERSARIAL_TRAINING_CONFIG,
            ...(config || {})
        };

        const adversarialTrainingSwitch = document.getElementById("adversarialTrainingSwitch");
        if (!adversarialTrainingSwitch) return;

        adversarialTrainingSwitch.checked = Boolean(adversarialTrainingConfig.enabled);
        setValue("adversarialTrainingDomain", adversarialTrainingConfig.domain);
        setValue("adversarialTrainingAttack", adversarialTrainingConfig.attack);
        setValue("adversarialTrainingEpsilon", adversarialTrainingConfig.epsilon);
        setValue("adversarialTrainingAlpha", adversarialTrainingConfig.alpha ?? "");
        setValue("adversarialTrainingSteps", adversarialTrainingConfig.steps);
        setValue("adversarialTrainingMode", adversarialTrainingConfig.mode);
        setValue("adversarialTrainingCleanWeight", adversarialTrainingConfig.clean_weight);
        setValue("adversarialTrainingAdversarialWeight", adversarialTrainingConfig.adversarial_weight);
        setValue("adversarialTrainingApplyProbability", adversarialTrainingConfig.apply_probability);
        setValue("adversarialTrainingClipMin", adversarialTrainingConfig.clip_min);
        setValue("adversarialTrainingClipMax", adversarialTrainingConfig.clip_max);

        const logMetricsInput = document.getElementById("adversarialTrainingLogMetrics");
        if (logMetricsInput) {
            logMetricsInput.checked = Boolean(adversarialTrainingConfig.log_adversarial_metrics);
        }

        toggleAdversarialTrainingSettings(adversarialTrainingSwitch.checked);
        updateDatasetAvailability();
    }

    function setValue(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.value = value;
        }
    }

    function resetAdversarialTrainingConfig() {
        setAdversarialTrainingConfig(DEFAULT_ADVERSARIAL_TRAINING_CONFIG);
    }

    function validateConfig() {
        const config = getAdversarialTrainingConfig();
        if (!config.enabled) {
            return null;
        }
        if (config.epsilon < 0) {
            return "[Adversarial Training] Epsilon must be greater than or equal to 0.";
        }
        if (config.attack === "pgd" && config.steps < 1) {
            return "[Adversarial Training] PGD steps must be at least 1.";
        }
        if (config.clean_weight < 0 || config.adversarial_weight < 0) {
            return "[Adversarial Training] Loss weights must be greater than or equal to 0.";
        }
        if (config.mode === "mixed" && config.clean_weight + config.adversarial_weight === 0) {
            return "[Adversarial Training] Mixed mode needs at least one positive loss weight.";
        }
        if (config.apply_probability < 0 || config.apply_probability > 1) {
            return "[Adversarial Training] Apply probability must be between 0 and 1.";
        }
        if (config.clip_min >= config.clip_max) {
            return "[Adversarial Training] Min bound must be smaller than max bound.";
        }
        return null;
    }

    return {
        initializeAdversarialTraining,
        getAdversarialTrainingConfig,
        setAdversarialTrainingConfig,
        resetAdversarialTrainingConfig,
        validateConfig
    };
})();

export default AdversarialTrainingManager;
