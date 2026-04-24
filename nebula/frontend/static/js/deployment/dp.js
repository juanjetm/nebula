// Differential Privacy Module
const DpManager = (function() {
    const DEFAULT_DP_CONFIG = {
        enabled: false
    };

    function initializeDifferentialPrivacy() {
        setupDpSwitch();
        setDpConfig(DEFAULT_DP_CONFIG);
    }

    function setupDpSwitch() {
        const dpSwitch = document.getElementById("dpSwitch");
        if (!dpSwitch) return;

        dpSwitch.addEventListener("change", function() {
            toggleDpSettings(this.checked);
        });
    }

    function toggleDpSettings(enabled) {
        const dpSettings = document.getElementById("dp-settings");
        if (!dpSettings) return;

        dpSettings.style.display = enabled ? "block" : "none";
    }

    function getDpConfig() {
        return {
            enabled: Boolean(document.getElementById("dpSwitch")?.checked)
        };
    }

    function setDpConfig(config = DEFAULT_DP_CONFIG) {
        const dpConfig = {
            ...DEFAULT_DP_CONFIG,
            ...(config || {})
        };

        const dpSwitch = document.getElementById("dpSwitch");
        if (!dpSwitch) return;

        dpSwitch.checked = Boolean(dpConfig.enabled);
        toggleDpSettings(dpSwitch.checked);
    }

    function resetDpConfig() {
        setDpConfig(DEFAULT_DP_CONFIG);
    }

    return {
        initializeDifferentialPrivacy,
        getDpConfig,
        setDpConfig,
        resetDpConfig
    };
})();

export default DpManager;
