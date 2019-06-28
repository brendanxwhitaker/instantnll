#! /usr/bin/env python

"""Script that runs all verification steps.
"""

import argparse
import os
import shutil
from subprocess import run
from subprocess import CalledProcessError
import sys

def main(checks):
    try:
        print("Verifying with " + str(checks))
        if "pytest" in checks:
            print("Tests (pytest):", flush=True)
            run("pytest -v --color=yes", shell=True, check=True)

        if "pylint" in checks:
            print("Linter (pylint):", flush=True)
            run("pylint -d locally-disabled,locally-enabled -f colorized instantnll", shell=True, check=True)
            print("pylint checks passed")

        if "mypy" in checks:
            print("Typechecker (mypy):", flush=True)
            run("mypy instantnll --ignore-missing-imports", shell=True, check=True)
            print("mypy checks passed")

        if "check-large-files" in checks:
            print("Checking all added files have size <= 2MB", flush=True)
            run("./scripts/check_large_files.sh 2", shell=True, check=True)
            print("check large files passed")

    except CalledProcessError:
        # squelch the exception stacktrace
        sys.exit(1)

if __name__ == "__main__":
    checks = ['pytest', 'pylint', 'mypy', 'build-docs', 'check-docs', 'check-links', 'check-requirements',
              'check-large-files']

    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', default=checks, nargs='+', choices=checks)

    args = parser.parse_args()

    main(args.checks)
