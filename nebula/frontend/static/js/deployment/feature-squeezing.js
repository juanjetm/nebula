// Feature Squeezing Module
const FeatureSqueezingManager = (function() {
    const DEFAULT_FEATURE_SQUEEZING_CONFIG = {
        enabled: false,
        bit_depth: 4
    };

    function initializeFeatureSqueezing() {
        setupFeatureSqueezingSwitch();
        setFeatureSqueezingConfig(DEFAULT_FEATURE_SQUEEZING_CONFIG);
    }

    function setupFeatureSqueezingSwitch() {
        const featureSqueezingSwitch = document.getElementById("featureSqueezingSwitch");
        if (!featureSqueezingSwitch) return;

        featureSqueezingSwitch.addEventListener("change", function() {
            toggleFeatureSqueezingSettings(this.checked);
        });
    }

    function toggleFeatureSqueezingSettings(enabled) {
        const featureSqueezingSettings = document.getElementById("feature-squeezing-settings");
        if (!featureSqueezingSettings) return;

        featureSqueezingSettings.style.display = enabled ? "block" : "none";
    }

    function getFeatureSqueezingConfig() {
        const nInput = document.getElementById("featureSqueezingN");
        const bitDepth = parseInt(nInput?.value, 10);

        return {
            enabled: Boolean(document.getElementById("featureSqueezingSwitch")?.checked),
            bit_depth: Number.isFinite(bitDepth)
                ? bitDepth
                : DEFAULT_FEATURE_SQUEEZING_CONFIG.bit_depth
        };
    }

    function setFeatureSqueezingConfig(config = DEFAULT_FEATURE_SQUEEZING_CONFIG) {
        const featureSqueezingConfig = {
            ...DEFAULT_FEATURE_SQUEEZING_CONFIG,
            ...(config || {})
        };
        const bitDepth = featureSqueezingConfig.bit_depth ?? featureSqueezingConfig.n;

        const featureSqueezingSwitch = document.getElementById("featureSqueezingSwitch");
        if (!featureSqueezingSwitch) return;

        featureSqueezingSwitch.checked = Boolean(featureSqueezingConfig.enabled);
        const nInput = document.getElementById("featureSqueezingN");
        if (nInput) {
            nInput.value = bitDepth ?? DEFAULT_FEATURE_SQUEEZING_CONFIG.bit_depth;
        }
        toggleFeatureSqueezingSettings(featureSqueezingSwitch.checked);
    }

    function resetFeatureSqueezingConfig() {
        setFeatureSqueezingConfig(DEFAULT_FEATURE_SQUEEZING_CONFIG);
    }

    return {
        initializeFeatureSqueezing,
        getFeatureSqueezingConfig,
        setFeatureSqueezingConfig,
        resetFeatureSqueezingConfig
    };
})();

export default FeatureSqueezingManager;
