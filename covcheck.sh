#!/bin/bash
pytest --cov=./
bash <(curl -s https://codecov.io/bash)
