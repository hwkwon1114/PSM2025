module load singularity 2>/dev/null
cd /projects/p33082/PSM2025-zigzag/docs/slides
singularity exec -B /projects:/projects /projects/p33082/.sing_cache/lo.sif env HOME=/tmp soffice --headless --convert-to pdf --outdir _render calibration_framework_slide.pptx
