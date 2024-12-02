python -u fine_tuning_test.py \
--plms --outdir results/cartoon/pretrained_model \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/cartoon/image \
--mask_folder dataset/testdata/cartoon/mask \
--reference_folder dataset/testdata/cartoon/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/handmake/pretrained_model \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/handmake/image \
--mask_folder dataset/testdata/handmake/mask \
--reference_folder dataset/testdata/handmake/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/painting/pretrained_model \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/painting/image \
--mask_folder dataset/testdata/painting/mask \
--reference_folder dataset/testdata/painting/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/sketch/pretrained_model \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/sketch/image \
--mask_folder dataset/testdata/sketch/mask \
--reference_folder dataset/testdata/sketch/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/tattoo/pretrained_model \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/tattoo/image \
--mask_folder dataset/testdata/tattoo/mask \
--reference_folder dataset/testdata/tattoo/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/weather/pretrained_model \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_folder dataset/testdata/weather/image \
--mask_folder dataset/testdata/weather/mask \
--reference_folder dataset/testdata/weather/reference \
--seed 321 \
--scale 5




python -u fine_tuning_test.py \
--plms --outdir results/cartoon/finetuned_model \
--config configs/v1.yaml \
--ckpt checkpoints/cartoon/fine_tuning_model.ckpt \
--image_folder dataset/testdata/cartoon/image \
--mask_folder dataset/testdata/cartoon/mask \
--reference_folder dataset/testdata/cartoon/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/handmake/finetuned_model \
--config configs/v1.yaml \
--ckpt checkpoints/handmake/fine_tuning_model.ckpt \
--image_folder dataset/testdata/handmake/image \
--mask_folder dataset/testdata/handmake/mask \
--reference_folder dataset/testdata/handmake/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/painting/finetuned_model \
--config configs/v1.yaml \
--ckpt checkpoints/painting/fine_tuning_model.ckpt \
--image_folder dataset/testdata/painting/image \
--mask_folder dataset/testdata/painting/mask \
--reference_folder dataset/testdata/painting/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/sketch/finetuned_model \
--config configs/v1.yaml \
--ckpt checkpoints/sketch/fine_tuning_model.ckpt \
--image_folder dataset/testdata/sketch/image \
--mask_folder dataset/testdata/sketch/mask \
--reference_folder dataset/testdata/sketch/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/tattoo/finetuned_model \
--config configs/v1.yaml \
--ckpt checkpoints/tattoo/fine_tuning_model.ckpt \
--image_folder dataset/testdata/tattoo/image \
--mask_folder dataset/testdata/tattoo/mask \
--reference_folder dataset/testdata/tattoo/reference \
--seed 321 \
--scale 5

python -u fine_tuning_test.py \
--plms --outdir results/weather/finetuned_model \
--config configs/v1.yaml \
--ckpt checkpoints/weather/fine_tuning_model.ckpt \
--image_folder dataset/testdata/weather/image \
--mask_folder dataset/testdata/weather/mask \
--reference_folder dataset/testdata/weather/reference \
--seed 321 \
--scale 5

