path="4 дз/Trabsformer"

dvc stage add -f \
    --name train \
    --deps "${path}"/conf \
    --deps "${path}"/train.py \
    --deps "${path}"/modules \
    --outs "${path}"/weights \
    python "${path}"/train.py