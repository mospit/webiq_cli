import logging

logger = logging.getLogger(__name__)

def get_stagehand_config(gemini_api_key: str, steel_api_key: str = "DUMMY_STEEL_KEY", steel_session_id: str = "dummy_steel_session_id"):
    """Returns the Stagehand configuration dictionary."""
    if not gemini_api_key:
        # In a real scenario, this might come from a config file or env variable
        gemini_api_key = "DUMMY_GEMINI_API_KEY"
        logger.warning("Using dummy Gemini API key for Stagehand configuration.")

    config = {
        "env": "LOCAL",
        "localBrowserLaunchOptions": {
            "cdpUrl": f"wss://connect.steel.dev?apiKey={steel_api_key}&sessionId={steel_session_id}"
        },
        "modelName": "google/gemini-2.0-flash",
        "modelClientOptions": {
            "apiKey": gemini_api_key,
            "maxTokens": 8192,
            "temperature": 0.1
        },
        "enableCaching": True,
        "verbose": 1
    }
    logger.info(f"Generated Stagehand config with Steel session: {steel_session_id}")
    return config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(get_stagehand_config("test_gemini_key", "test_steel_key", "test_session_id"))
