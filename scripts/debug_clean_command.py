import asyncio

from pff.infrastructure.cleanup.commands.database import KGDataCleanCommand


async def test_command():
    cmd = KGDataCleanCommand()
    print(f"Testing {cmd.label}")

    try:
        preview = await cmd.get_preview()
        print(f"Preview result: {preview}")
    except Exception as e:
        print(f"Preview crashed: {e}")


if __name__ == "__main__":
    asyncio.run(test_command())
