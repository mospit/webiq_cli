import unittest
from unittest import mock
from click.testing import CliRunner
from webiq.webiq.cli.main import webiq # Main CLI application
import asyncio

# Helper to run async functions in mock
async def async_mock_return(value):
    return value

class MockAsyncFunction:
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.call_args_list = []

    async def __call__(self, *args, **kwargs):
        self.call_args_list.append(mock.call(*args, **kwargs))
        return self.return_value

class TestRecordCommand(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    @mock.patch('webiq.webiq.cli.main.goal_aware_recording') # Path to where goal_aware_recording is USED
    def test_record_command_basic(self, mock_goal_aware_recording):
        # Configure the mock session and its methods
        mock_session = mock.MagicMock()
        mock_session.goal = "test goal"
        mock_session.steel_session = {'id': 'sim_steel_session_123'}
        mock_session.goal_analysis = {'analysis': 'done'}
        mock_session.screenshots = ['shot1.png']
        mock_session.goal_progress = [{'progress': '50%'}]

        # Make start_monitoring an async mock
        mock_session.start_monitoring = MockAsyncFunction()

        # Configure goal_aware_recording to return a future resolving to mock_session
        mock_goal_aware_recording.return_value = asyncio.Future()
        mock_goal_aware_recording.return_value.set_result(mock_session)

        result = self.runner.invoke(webiq, ['record', 'http://example.com', 'test goal'])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Initiating recording session for URL: http://example.com with goal: 'test goal'", result.output)
        self.assertIn("Recording session successfully created for goal: 'test goal'", result.output)
        self.assertIn("Steel Session ID (simulated): sim_steel_session_123", result.output)
        self.assertIn("Starting recording monitoring...", result.output)
        self.assertIn("Recorded Screenshots (simulated): ['shot1.png']", result.output)

        mock_goal_aware_recording.assert_called_once()
        # Check basic args; detailed options check can be more specific
        args, kwargs = mock_goal_aware_recording.call_args
        self.assertEqual(kwargs['url'], 'http://example.com')
        self.assertEqual(kwargs['goal'], 'test goal')
        self.assertTrue(mock_session.start_monitoring.call_args_list)


    @mock.patch('webiq.webiq.cli.main.goal_aware_recording')
    def test_record_command_with_options(self, mock_goal_aware_recording):
        mock_session = mock.MagicMock()
        mock_session.goal = "test goal with options"
        mock_session.steel_session = {'id': 'sim_steel_session_456'}
        mock_session.goal_analysis = {'analysis': 'deep_done'}
        mock_session.screenshots = ['shot_opt.png']
        mock_session.goal_progress = [{'progress': '75%'}]
        mock_session.start_monitoring = MockAsyncFunction()

        mock_goal_aware_recording.return_value = asyncio.Future()
        mock_goal_aware_recording.return_value.set_result(mock_session)

        result = self.runner.invoke(webiq, [
            'record', 'http://options.com', 'test goal with options',
            '--session-name', 'my_session',
            '--timeout', '600',
            '--vision-mode',
            '--analyze-deep'
        ])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Initiating recording session for URL: http://options.com with goal: 'test goal with options'", result.output)
        self.assertIn("Session Name: my_session", result.output) # Check if session name is used in output
        # Timeout is used in options dict passed to goal_aware_recording, not directly in output unless we add it

        mock_goal_aware_recording.assert_called_once()
        args, kwargs = mock_goal_aware_recording.call_args
        self.assertEqual(kwargs['url'], 'http://options.com')
        self.assertEqual(kwargs['goal'], 'test goal with options')
        self.assertEqual(kwargs['options']['session_name'], 'my_session')
        self.assertEqual(kwargs['options']['timeout'], 600)
        self.assertTrue(kwargs['options']['vision_mode'])
        self.assertTrue(kwargs['options']['analyze_deep'])
        self.assertTrue(mock_session.start_monitoring.call_args_list)

    @mock.patch('webiq.webiq.cli.main.goal_aware_recording')
    def test_record_command_api_key_env_vars(self, mock_goal_aware_recording):
        mock_session = mock.MagicMock()
        mock_session.start_monitoring = MockAsyncFunction()
        mock_goal_aware_recording.return_value = asyncio.Future()
        mock_goal_aware_recording.return_value.set_result(mock_session)

        with mock.patch.dict('os.environ', {'GEMINI_API_KEY': 'env_gem_key', 'STEEL_API_KEY': 'env_steel_key'}):
            self.runner.invoke(webiq, ['record', 'http://env.com', 'env test'])

        args, kwargs = mock_goal_aware_recording.call_args
        self.assertEqual(kwargs['gemini_api_key'], 'env_gem_key')
        self.assertEqual(kwargs['steel_api_key'], 'env_steel_key')


if __name__ == '__main__':
    unittest.main()
