#!/usr/bin/env python3
"""Root launcher for the POPCROP Spot-Aware Manual Base Caller & Signal Validator."""

import os
import sys
import argparse
import webbrowser
import threading

# Add current workspace to path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from popcrop.annotation import app as app_module


def parse_args():
    parser = argparse.ArgumentParser(description="POPCROP Spot-Aware Manual Base Caller & Signal Validator")
    parser.add_argument("--image", "--tiff", dest="image", default=None, help="Path to multichannel TIFF or ND2 file")
    parser.add_argument("--tsv", default=None, help="Path to cell coordinates TSV/CSV file")
    parser.add_argument("--diameter", type=float, default=None, help="Estimated nuclear diameter in pixels")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open default browser")
    return parser.parse_args()


def open_browser(port: int):
    webbrowser.open(f"http://localhost:{port}")


def main():
    args = parse_args()
    if args.image:
        app_module.CUSTOM_TIFF_PATH = os.path.abspath(args.image)
    if args.tsv:
        app_module.ACTIVE_TSV_PATH = os.path.abspath(args.tsv)
    if args.diameter:
        app_module.DEFAULT_NUCLEI_DIAMETER = float(args.diameter)

    print("=================================================================")
    print("  POPCROP Spot-Aware Manual Base Caller & Signal Validator")
    print("=================================================================")
    print(f"  Server URL:         http://localhost:{args.port}")
    print(f"  Base Caller:        http://localhost:{args.port}/base_caller")
    print(f"  Blob Validator:     http://localhost:{args.port}/blob_validator")
    print(f"  Base Ground Truth:  {app_module.BASE_GROUND_TRUTH_PATH}")
    print(f"  Blob Ground Truth:  {app_module.BLOB_GROUND_TRUTH_PATH}")
    print("=================================================================")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    app_module.app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
