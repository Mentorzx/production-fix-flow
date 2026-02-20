"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/core/verify_logging_pkg.py

"""

import os
import subprocess
import sys
from pathlib import Path


def run_test(name, code, env=None):
    """Execute run test.



    Args:

        name: Input value used by this callable.

        code: Input value used by this callable.

        env: Optional input value.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    print(f"Running {name}...", end=" ", flush=True)
    my_env = os.environ.copy()
    if env:
        my_env.update(env)

    try:
        subprocess.run(
            [sys.executable, "-c", code],
            env=my_env,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            check=True,
        )
        print("PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED")
        print("--- stdout ---")
        print(e.stdout)
        print("--- stderr ---")
        print(e.stderr)
        return False


def main():
    """Execute main.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    results = []

    # 1. Smoke Import
    results.append(
        run_test(
            "Smoke Import",
            """
import pff.shared.core.logging as pkg
print(f"Type of pkg: {type(pkg)}")
# print(f"Dir of pkg: {dir(pkg)}")
# If it is a Logger object, dir() works too.

if hasattr(pkg, 'logger'):
    print("pkg is the package")
    assert pkg.logger is not None
else:
    print("pkg MIGHT be the logger object itself?")
    # If pkg is the logger, it should have .info
    if hasattr(pkg, 'info'):
        print("pkg IS the logger object")
    else:
        print("pkg is unknown")

print("Import successful")
    """,
        )
    )

    # 2. Log Level Env
    results.append(
        run_test(
            "Log Level Env",
            """
import os
from pathlib import Path
import pff.shared.core.logging as pkg

# Use the exported logger object
log = pkg.logger if hasattr(pkg, 'logger') else pkg

log.warning("Warning visible")
log.info("Info hidden")
    """,
            env={"LOG_LEVEL": "WARNING", "DISABLE_RICH": "1"},
        )
    )

    # 3. File Writing
    tmp_dir = Path("tests/shared/core/temp_logs")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    code_file_write = f"""
import time
from pathlib import Path
import pff.shared.core.logging as pkg

log = pkg.logger if hasattr(pkg, 'logger') else pkg

log.info("Test file write")
log.complete()

# Wait for async write
time.sleep(1)

log_dir = Path("{tmp_dir}")
files = list(log_dir.glob("*.log"))
assert len(files) > 0
content = files[0].read_text()
assert "Test file write" in content
    """
    results.append(
        run_test(
            "File Write",
            code_file_write,
            env={"LOG_DIR": str(tmp_dir), "FILE_LOG_LEVEL": "INFO"},
        )
    )

    # 4. ContextVars / OTel
    results.append(
        run_test(
            "OTel Context",
            """
import pff.shared.core.logging as pkg
import orjson

log = pkg.logger if hasattr(pkg, 'logger') else pkg
start_span = pkg.start_span

with start_span("test_span") as span:
    log.info("Inside span")

log.complete()
    """,
        )
    )

    # 5. Utilities
    results.append(
        run_test(
            "Utilities",
            """
import pff.shared.core.logging as pkg

timeit = pkg.timeit
suppress_output = pkg.suppress_output

@timeit
def func():
    pass

func()

with suppress_output():
    print("Hidden")
    """,
        )
    )

    # Cleanup
    import shutil

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    if all(results):
        print("\nAll package verification tests PASSED.")
        sys.exit(0)
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
