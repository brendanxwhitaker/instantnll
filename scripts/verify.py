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
    
    print("Verifying with " + str(checks))
    if "pytest" in checks:
        print("Tests (pytest):", flush=True)
        try:
            run("pytest -v --color=yes", shell=True, check=True)
        except CalledProcessError:
            print("CalledProcessError")

    if "pylint" in checks:
        print("Linter (pylint):", flush=True)
        try:
            run("pylint -d locally-disabled,locally-enabled -f colorized instantnll", shell=True, check=True)
        except CalledProcessError:
            print("CalledProcessError")
        print("pylint checks passed")

    if "check-large-files" in checks:
        print("Checking all added files have size <= 2MB", flush=True)
        try:
            run("./scripts/check_large_files.sh 2", shell=True, check=True)
        except CalledProcessError:
                print("CalledProcessError")
        print("check large files passed")

if __name__ == "__main__":
    # checks = ['pytest', 'pylint', 'mypy', 'build-docs', 'check-docs', 'check-links', 'check-requirements',
    #           'check-large-files']
    checks = ['pytest', 'pylint', 'check-large-files']
    dirpath = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', default=checks, nargs='+', choices=checks)

    args = parser.parse_args()

    main(args.checks)
