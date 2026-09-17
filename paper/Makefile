# Build the manuscript. Requires pdflatex and bibtex; figures are built by
# scripts/14_make_figures.py from the files in results/.
.PHONY: all figures clean

all: paper.pdf

figures:
	cd .. && python scripts/14_make_figures.py

paper.pdf: paper.tex preamble.tex references.bib $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc
