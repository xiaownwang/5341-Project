python -u fine_tuning_test.py \
--plms --outdir results/cartoon \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/cartoon/image \
--mask_folder dataset/testdata/cartoon/mask \
--reference_folder dataset/testdata/cartoon/reference \
--seed 321 \
--scale 5


##################TODO########################
# add more code to test different type of images

