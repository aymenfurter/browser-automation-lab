"""Entry point for the Copilot SDK browser automation agent."""

import asyncio

from copilot_sdk_agent.agent import run_agent


def main():
    """Run the Copilot SDK agent."""
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
