PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: analysis_plan.pdf phase1_report.pdf phase2_report.pdf phase3_report.pdf

# Two passes so \ref / \label cross-references resolve.
%.pdf: %.tex
	$(PDFLATEX) $<
	$(PDFLATEX) $<

# The phase reports \input generated tables and figures, so a stale artefact is
# a build dependency rather than a transcription risk nobody notices.
phase1_report.pdf: phase1_report.tex $(wildcard figures/*.pdf)
phase2_report.pdf: phase2_report.tex $(wildcard tables/phase2_*.tex) $(wildcard figures/phase2_*.pdf)
phase3_report.pdf: phase3_report.tex $(wildcard tables/phase3_*.tex) $(wildcard figures/phase3_*.pdf)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg
