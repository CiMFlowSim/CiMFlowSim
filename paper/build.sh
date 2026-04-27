#!/bin/bash
export PATH="/usr/local/texlive/2025/bin/x86_64-linux:$PATH"
export TEXINPUTS=".:./template:"
export BSTINPUTS=".:./template:"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
