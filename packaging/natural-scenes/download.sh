#!/bin/bash
aws s3 sync s3://natural-scenes-dataset/ "$SHARED_DATASETS/natural-scenes" \
--exclude "*" \
--include "*ppdata*func1pt8mm*streams.nii.gz" \
--include "*ppdata*func1pt8mm*prf-visualrois.nii.gz" \
--include "*ppdata*func1pt8mm*prf-eccrois.nii.gz" \
--include "*ppdata*func1pt8mm*floc-faces.nii.gz" \
--include "*ppdata*func1pt8mm*floc-bodies.nii.gz" \
--include "*ppdata*func1pt8mm*floc-places.nii.gz" \
--include "*ppdata*func1pt8mm*floc-words.nii.gz" \
--include "*nsddata/freesurfer/subj*/label/streams.mgz.ctab" \
--include "*nsddata/freesurfer/subj*/label/prf-visualrois.mgz.ctab" \
--include "*nsddata/freesurfer/subj*/label/prf-eccrois.mgz.ctab" \
--include "*nsddata/freesurfer/subj*/label/floc-faces.mgz.ctab" \
--include "*nsddata/freesurfer/subj*/label/floc-bodies.mgz.ctab" \
--include "*nsddata/freesurfer/subj*/label/floc-places.mgz.ctab" \
--include "*nsddata/freesurfer/subj*/label/floc-words.mgz.ctab" \
--include "*ppdata*behav/responses.tsv" \
--include "*nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5" \
--include "*experiments/nsd/nsd_stim_info_merged.csv" \
--include "*stimuli/nsd/shared1000.tsv" \
--include "*stimuli/nsd/special100.tsv" \
--include "*stimuli/nsd/special3.tsv" \
--include "*stimuli/nsd/notshown.tsv" \
--include "*nsddata_betas/ppdata/subj*/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session*.hdf5" \
--include "*nsddata_betas/ppdata/subj*/func1pt8mm/betas_fithrf_GLMdenoise_RR/ncsnr*.nii.gz"
