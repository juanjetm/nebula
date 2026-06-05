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
        apply_probability: 0.3,
        log_adversarial_metrics: true,
        target_loss_increase: null,
        max_loss_increase: null
    };

    const IMAGE_DATASETS = new Set(["MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "CIFAR100"]);
    const TABULAR_ADVERSARIAL_DATASETS = new Set(["AdultCensus"]);
    const IMAGE_ATTACK_OPTIONS = [
        {value: "fgsm", label: "FGSM"},
        {value: "pgd", label: "PGD"}
    ];
    const TABULAR_ATTACK_OPTIONS = [
        {value: "capgd", label: "CAPGD"}
    ];

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
        const stepsTitle = document.getElementById("adversarialTrainingStepsTitle");
        const lossWindowSettings = document.getElementById("adversarial-training-loss-window-settings");
        const domain = document.getElementById("adversarialTrainingDomain")?.value || DEFAULT_ADVERSARIAL_TRAINING_CONFIG.domain;
        if (!pgdSettings) return;

        pgdSettings.style.display = ["pgd", "capgd"].includes(attack) ? "block" : "none";
        if (lossWindowSettings) {
            lossWindowSettings.style.display = domain === "tabular" ? "block" : "none";
        }
        if (stepsTitle) {
            stepsTitle.textContent = domain === "tabular" ? "CAPGD steps" : "PGD steps";
        }
    }

    function updateDatasetAvailability() {
        const dataset = document.getElementById("datasetSelect")?.value;
        const domain = getDatasetDomain(dataset);
        const adversarialTrainingSwitch = document.getElementById("adversarialTrainingSwitch");
        const datasetNote = document.getElementById("adversarial-training-dataset-note");
        const domainInput = document.getElementById("adversarialTrainingDomain");
        const settings = document.getElementById("adversarial-training-settings");

        if (datasetNote) {
            datasetNote.style.display = domain === "unsupported" ? "block" : "none";
            datasetNote.textContent = "Adversarial Training for tabular datasets currently supports AdultCensus with CAPGD.";
        }
        if (domainInput) {
            domainInput.value = domain === "unsupported" ? "tabular" : domain;
        }

        if (!adversarialTrainingSwitch) return;
        adversarialTrainingSwitch.disabled = domain === "unsupported";
        if (domain === "unsupported") {
            adversarialTrainingSwitch.checked = false;
            if (settings) {
                settings.style.display = "none";
            }
            return;
        }

        adversarialTrainingSwitch.disabled = false;
        refreshAttackOptions(domain);
        toggleAdversarialTrainingSettings(adversarialTrainingSwitch.checked);
    }

    function getDatasetDomain(dataset) {
        if (IMAGE_DATASETS.has(dataset)) {
            return "image";
        }
        if (TABULAR_ADVERSARIAL_DATASETS.has(dataset)) {
            return "tabular";
        }
        return "unsupported";
    }

    function refreshAttackOptions(domain, preferredAttack = null) {
        const attackSelect = document.getElementById("adversarialTrainingAttack");
        if (!attackSelect) return;

        // Tabular datasets intentionally expose only CAPGD; image datasets expose FGSM/PGD.
        const options = domain === "tabular" ? TABULAR_ATTACK_OPTIONS : IMAGE_ATTACK_OPTIONS;
        const currentAttack = preferredAttack || attackSelect.value;
        attackSelect.innerHTML = "";
        options.forEach(({value, label}) => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            attackSelect.appendChild(option);
        });

        const validAttack = options.some(option => option.value === currentAttack)
            ? currentAttack
            : options[0].value;
        attackSelect.value = validAttack;
        attackSelect.disabled = domain === "tabular";
        toggleAttackSettings(validAttack);
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
        const domain = document.getElementById("adversarialTrainingDomain")?.value || DEFAULT_ADVERSARIAL_TRAINING_CONFIG.domain;
        const attack = domain === "tabular"
            ? "capgd"
            : (document.getElementById("adversarialTrainingAttack")?.value || DEFAULT_ADVERSARIAL_TRAINING_CONFIG.attack);
        const config = {
            enabled: Boolean(document.getElementById("adversarialTrainingSwitch")?.checked),
            domain,
            attack,
            epsilon: numberValue("adversarialTrainingEpsilon", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.epsilon),
            alpha: optionalNumberValue("adversarialTrainingAlpha", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.alpha),
            steps: integerValue("adversarialTrainingSteps", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.steps),
            mode: document.getElementById("adversarialTrainingMode")?.value || DEFAULT_ADVERSARIAL_TRAINING_CONFIG.mode,
            apply_probability: numberValue("adversarialTrainingApplyProbability", DEFAULT_ADVERSARIAL_TRAINING_CONFIG.apply_probability),
            target_loss_increase: optionalNumberValue(
                "adversarialTrainingTargetLossIncrease",
                DEFAULT_ADVERSARIAL_TRAINING_CONFIG.target_loss_increase
            ),
            max_loss_increase: optionalNumberValue(
                "adversarialTrainingMaxLossIncrease",
                DEFAULT_ADVERSARIAL_TRAINING_CONFIG.max_loss_increase
            ),
            log_adversarial_metrics: true
        };

        if (config.alpha === null || config.attack !== "pgd") {
            delete config.alpha;
        }
        if (config.target_loss_increase === null) {
            delete config.target_loss_increase;
        }
        if (config.max_loss_increase === null) {
            delete config.max_loss_increase;
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
        setValue("adversarialTrainingEpsilon", adversarialTrainingConfig.epsilon);
        setValue("adversarialTrainingAlpha", adversarialTrainingConfig.alpha ?? "");
        setValue("adversarialTrainingSteps", adversarialTrainingConfig.steps);
        setValue(
            "adversarialTrainingMode",
            ["mixed", "adversarial"].includes(adversarialTrainingConfig.mode)
                ? adversarialTrainingConfig.mode
                : DEFAULT_ADVERSARIAL_TRAINING_CONFIG.mode
        );
        setValue("adversarialTrainingApplyProbability", adversarialTrainingConfig.apply_probability);
        setValue("adversarialTrainingTargetLossIncrease", adversarialTrainingConfig.target_loss_increase ?? "");
        setValue("adversarialTrainingMaxLossIncrease", adversarialTrainingConfig.max_loss_increase ?? "");

        updateDatasetAvailability();
        const domain = document.getElementById("adversarialTrainingDomain")?.value || adversarialTrainingConfig.domain;
        refreshAttackOptions(domain, adversarialTrainingConfig.attack);
        toggleAdversarialTrainingSettings(adversarialTrainingSwitch.checked);
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
        if (["pgd", "capgd"].includes(config.attack) && config.steps < 1) {
            return "[Adversarial Training] Search steps must be at least 1.";
        }
        if (!["mixed", "adversarial"].includes(config.mode)) {
            return "[Adversarial Training] Training mode must be Clean + adversarial or Adversarial only.";
        }
        if (config.apply_probability < 0 || config.apply_probability > 1) {
            return "[Adversarial Training] Apply probability must be between 0 and 1.";
        }
        if (config.target_loss_increase !== undefined && config.target_loss_increase < 0) {
            return "[Adversarial Training] Target loss increase must be greater than or equal to 0.";
        }
        if (config.max_loss_increase !== undefined && config.max_loss_increase < 0) {
            return "[Adversarial Training] Max loss increase must be greater than or equal to 0.";
        }
        if (
            config.target_loss_increase !== undefined
            && config.max_loss_increase !== undefined
            && config.target_loss_increase > config.max_loss_increase
        ) {
            return "[Adversarial Training] Target loss increase must be smaller than or equal to max loss increase.";
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
