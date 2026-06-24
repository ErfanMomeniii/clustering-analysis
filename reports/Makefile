PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean analysis_plan phase1_report

all: analysis_plan.pdf phase1_report.pdf

analysis_plan.pdf: analysis_plan.tex
	$(PDFLATEX) analysis_plan.tex
	$(PDFLATEX) analysis_plan.tex

phase1_report.pdf: phase1_report.tex
	$(PDFLATEX) phase1_report.tex
	$(PDFLATEX) phase1_report.tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg
