docker run --gpus=all -v C:\Users\Public\Documents\tanagraspace\data\jpg:/input -v  C:\Users\Public\Documents\tanagraspace\opssat-sar-ai\tools\ml\output:/output -v  C:\Users\Public\Documents\tanagraspace\opssat-sar-ai\tools\ml\learn.py:/scripts/learn.py -v  C:\Users\Public\Documents\tanagraspace\opssat-sar-ai\tools\windows\run.sh:/scripts/run.sh -it --rm --name opssat-sar-ai public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tflite_model_maker:2.4.3
