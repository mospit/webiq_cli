import click
import asyncio
import os
import logging
from webiq.core.automation_service import goal_aware_recording # Updated import

# Configure logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def webiq():
    """AI-Powered Web Automation CLI Tool"""
    pass

@webiq.command()
@click.argument('url')
@click.argument('goal')
@click.option('--vision-mode', is_flag=True, help='Enable enhanced visual analysis.')
@click.option('--analyze-deep', is_flag=True, help='Perform comprehensive page analysis.')
@click.option('--session-name', help='Custom name for the recording session.')
@click.option('--timeout', default=1800, type=int, help='Maximum recording duration (default: 1800s).')
def record(url: str, goal: str, vision_mode: bool, analyze_deep: bool, session_name: str, timeout: int):
    """Record user interactions while understanding the goal."""
    logger.info(f"Initiating recording session for URL: {url} with goal: '{goal}'")

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_PLACEHOLDER")
    steel_api_key = os.environ.get("STEEL_API_KEY", "YOUR_STEEL_API_KEY_PLACEHOLDER")

    if gemini_api_key == "YOUR_GEMINI_API_KEY_PLACEHOLDER":
        logger.warning("Gemini API key not found in environment variables. Using placeholder.")
    if steel_api_key == "YOUR_STEEL_API_KEY_PLACEHOLDER":
        logger.warning("Steel API key not found in environment variables. Using placeholder.")

    options = {
        "vision_mode": vision_mode,
        "analyze_deep": analyze_deep,
        "session_name": session_name,
        "timeout": timeout
    }

    async def def_record_async():
        try:
            logger.info("Starting asynchronous recording process...")
            session = await goal_aware_recording(
                url=url,
                goal=goal,
                options=options,
                gemini_api_key=gemini_api_key,
                steel_api_key=steel_api_key
            )

            if session:
                logger.info(f"Recording session successfully created for goal: '{session.goal}'. Steel Session ID (simulated): {session.steel_session.get('id')}")
                click.echo(f"Goal analysis: {session.goal_analysis}")

                click.echo("Starting recording monitoring...")
                await session.start_monitoring() # This will run for a few simulated iterations

                click.echo("Recording finished.")
                click.echo(f"Session Name: {session_name if session_name else 'N/A'}")
                click.echo(f"Recorded Screenshots (simulated): {session.screenshots}")
                click.echo(f"Goal Progress Log (simulated): {session.goal_progress}")

                # In a real application, you might save this session data to a file
                # For now, just printing to console
            else:
                logger.error("Failed to create recording session.")
                click.echo("Error: Could not initialize recording session.", err=True)

        except Exception as e:
            logger.error(f"An error occurred during the recording process: {e}", exc_info=True)
            click.echo(f"An error occurred: {e}", err=True)

    asyncio.run(def_record_async())

if __name__ == '__main__':
    webiq()
