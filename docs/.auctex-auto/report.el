;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("pgfplots" "") ("listings" "") ("tabularx" "") ("svg" "inkscapelatex=false") ("inputenc" "utf8") ("fontenc" "T1") ("babel" "") ("xcolor" "") ("graphicx" "") ("geometry" "top=0.5in" "bottom=0.5in" "right=0.5in" "left=0.5in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "pgfplots"
    "graphicx"
    "listings"
    "tabularx"
    "svg"
    "inputenc"
    "fontenc"
    "babel"
    "xcolor"
    "geometry")
   (LaTeX-add-labels
    "tab:features"
    "tab:class_distribution"
    "fig:distributions"
    "fig:scaled_distributions"
    "fig:app"))
 :latex)

