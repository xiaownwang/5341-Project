python -u fine_tuning_test.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/image \
--mask_folder dataset/testdata/mask \
--reference_folder dataset/testdata/reference \
--seed 321 \
--scale 5


##################TODO########################
# add more code to test different type of images

