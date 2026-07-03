Here is the breakdown of exactly which uploaded file generates the data and visuals for the specific figures in the paper:



**Python Scripts (Mathematical Modeling \& Direct Visuals)**

**model.py**: Generates the time-series line graphs tracking how the sensitive and resistant cell populations change over time.



Figures: 2C, 2D (Constant Dose Therapy dynamics), 3C, 3D (Adaptive Therapy dynamics), and Supplementary Figure S3.



**draw\_heatmap.py**: Generates the 2D color maps visualizing the steady-state outcomes.



Figure: 1A (Fraction of sensitive cells based on varying combinations of competition and transition rates).



**sensitivity\_analysis.py**: This script acts as the engine to execute the large-scale parameter sweeps (the Latin Hypercube and log-uniform sampling of 10,000 combinations) across the biological bounds. It generates the raw datasets that the R scripts rely on, rather than plotting a specific final figure.



**utils.py**: A structural helper script containing background functions (like params\_to\_text and file-naming logic) used by model.py and draw\_heatmap.py to label and save the plots correctly.



**R Markdown Scripts (Statistical Analyses \& Classifications)**

**CDT-sims-uniform-sampling-50-50.Rmd**: Processes the datasets for Constant Dose Therapy (CDT). It categorizes the endpoints, calculates the proportion of "Favourable" vs. "Unfavourable" outcomes, and computes the statistical odds ratios.



Figures: 2A, 2B (It also likely calculates the baseline "no therapy" comparisons for 1B, 1C, 1D, and generates the bar charts in Supplementary Figures S1 \& S2).



**AT-sims-combined-sampling-50-50.rmd**: Processes the datasets for Adaptive Therapy (AT). It identifies which parameter combinations successfully lead to stable population cycles, calculates the period of those oscillations, and evaluates the effect sizes (Cohen's d).



Figures: 3A, 3B (Outcome proportions and odds ratios for AT), and 4A, 4B, 4C (Period distributions, effect sizes, and violin plots of stable population cycles).



**AT-virtual-cohorts.Rmd**: Handles the data for the "delayed treatment" scenarios, which simulate patients presenting at the clinic with varying initial tumor sizes. It specifically computes the Principal Component Analyses (PCAs).



Figures: 5A, 5B, 5C (Virtual cohort outcomes and PCA biplots).





1\. **Cytotoxic Adaptive Therapy (Cell Killing)**

To simulate a drug that actively kills cells while cycling on and off:



Keep r\_s\_treatment equal to r\_s (the drug doesn't change the growth rate).



Increase d\_s\_treatment to a value greater than 0 (the drug kills the cells).



Set your on/off thresholds (e.g., threshold\_treatment\_on = 0.5 and threshold\_treatment\_off = 0.25).



2\. **Cytostatic Adaptive Therapy (Growth Inhibiting)**

To simulate a drug that halts cell division but doesn't explicitly kill them:



Decrease r\_s\_treatment to a value much lower than r\_s (the drug severely slows growth).



Set d\_s\_treatment = 0 (the drug does not actively kill cells).



Set your on/off thresholds exactly as you did for the cytotoxic scenario.



3\. **Constant Dose Therapy (CDT)**

To simulate a standard, non-adaptive treatment where the patient receives the drug continuously:



Set threshold\_treatment\_on = 0 (therapy turns on the moment the simulation starts).



Set threshold\_treatment\_off = np.nan or a negative number (the population will never drop below this threshold, so the therapy never turns off).



You can then adjust d\_s\_treatment or r\_s\_treatment to make this continuous dose either cytotoxic or cytostatic.


Now, maybe calculate Time to Progression(TTP)

since TTP favors CDT over AT for symmetric competition(c =d = 0.2), we can try for various types of competition.
Other thing that can be done is varying the treatment_threshold_off from 0.1K to 0.4K. This is for checking stable cycles.
Whatever results we obtained, there should be a plot or verbal explanation convincing enough.
if Phase Space Plots aren't feasible, then time series plots can be used, like in this case.