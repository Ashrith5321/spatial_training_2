import sys
import os
from collections import namedtuple

# ==========================================
# PART 1: The Version Spoof (Must run first)
# ==========================================

# Define the target version we want to mimic (3.10.12)
# We must use a named tuple because Ray accesses attributes like .major
VersionInfo = namedtuple("VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"])
TARGET_TUPLE = VersionInfo(3, 10, 12, "final", 0)
TARGET_STR = "3.10.12 (main, Spoofed Date, 00:00:00) [GCC 1.2.3]"

print(f"DEBUG: Spoofing active. Original: {sys.version.split()[0]}", file=sys.stderr)

try:
    # 1. Patch sys.version_info (The struct)
    sys.version_info = TARGET_TUPLE

    # 2. Patch sys.version (The string)
    # We replace the version number in the string
    if "3.10.18" in sys.version:
        sys.version = sys.version.replace("3.10.18", "3.10.12")
    
    # 3. Patch platform.python_version (Fallback used by some libs)
    import platform
    platform.python_version = lambda: "3.10.12"

    print(f"DEBUG: Spoofed as: {sys.version_info}", file=sys.stderr)

except Exception as e:
    print(f"!! SPOOF ERROR: {e}", file=sys.stderr)

# ==========================================
# PART 2: The CLI Proxy
# ==========================================

# Ray's CLI is built with 'click'. We import the main command group.
from ray.scripts.scripts import cli

if __name__ == "__main__":
    # FIX: "fake_ray doesn't recognize --address"
    # Click uses sys.argv[0] to determine the command name.
    # We must trick it into thinking it's running as 'ray'.
    sys.argv[0] = "ray"

    # Now we invoke the CLI. It will automatically read sys.argv[1:]
    # which contains 'start', '--address', etc.
    try:
        cli()
    except SystemExit as e:
        # Pass the exit code back to the shell (e.g., 0 for success, 1 for error)
        sys.exit(e.code)
    except Exception as e:
        # Catch other crashes
        print(f"!! WRAPPER CRASH: {e}", file=sys.stderr)
        sys.exit(1)