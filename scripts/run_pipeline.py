import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.e2e.test_pipeline import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())