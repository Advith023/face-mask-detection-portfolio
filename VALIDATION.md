# Validation and implementation status

## Reproduced image-level benchmark

On 17 November 2026, the benchmark ran locally on an Apple Silicon Mac
using Python 3.9.6 and the recorded dependency environment.

The run processed 3,833 images, removed 85 byte-identical duplicate
copies and evaluated 3,748 images. Each image has three prediction
records, giving 11,244 records. Aggregate counts were reproduced exactly.

Run the results consistency check without installing ML dependencies:

    python3 check_results.py

This checks record uniqueness, image identity across configurations,
configuration coverage and agreement with the aggregate CSV.
It does not run inference or authenticate the original image files.

## Interpretation

Scoring uses folder labels and one prediction per image, based on the
highest-confidence valid face detection. No prediction counts as a miss.
This measures image-level label agreement, not per-face detection accuracy.

The original classifier training data and training split are unverified.
No byte-identical matches were found against the supplied recovered
dataset. This does not exclude transformed duplicates or prove that
evaluation images were absent from the original training data.

The reported improvement changes both detection threshold and crop
padding; the image benchmark does not isolate their individual effects.

## Recovered per-face evaluation

results/recovered-validation.csv contains historical per-face evaluation
results. The evaluation script, annotation manifest, split definition
and matching protocol are not currently published. This evaluation has
not been reproduced through the current repository.

Image-level and per-face results use different scoring units and datasets
and should not be compared as interchangeable accuracy measurements.

## Archived modules

The original/ directory contains historical source. The current installation
instructions and dependency pins support predict_image.py and
benchmark_images.py, not every archived script.

Archived video inference needs input-batch corrections and runtime testing.
Hardware code requires its original serial configuration and hardware.
Arduino firmware and wiring instructions are not currently supplied.

The archived social-distance prototype uses pixel distances without camera
calibration and evaluates separation only when exactly two faces are detected.
Its displayed centimetre values are not validated physical measurements.

The desktop interface, email alerts, hardware integration and social-distance
prototype have not been revalidated as an integrated system.

## Model availability

The model manifest identifies the files used in the successful local run.
The files must currently be supplied locally; an authorised public acquisition
route for the exact classifier remains unresolved.

The benchmark results do not establish suitability for medical diagnosis
or autonomous access-control decisions.
