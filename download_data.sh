cd keypoint_files

conda deactivate

# download deft_shrimp data
gsutil -m cp -r \
  "gs://mediapipe-data/neuro_clinic_may_16/deft_shrimp" \
  .
# download tuned_goose data
gsutil -m cp -r \
  "gs://mediapipe-data/neuro_clinic_may_16/tuned_goose" \
  .

# download patient videos data
gsutil -m cp -r \
  "gs://mediapipe-data/neuro_clinic_may_16/patient_videos" \
  .
  
cd ..

conda activate tf2