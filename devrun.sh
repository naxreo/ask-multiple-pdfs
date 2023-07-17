#!/bin/bash

docker run -it --rm --name ai -v $(pwd):/app -p 8501:8501 python:3.10.12-bullseye bash
