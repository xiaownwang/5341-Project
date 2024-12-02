python fine_tuning_test_metric.py \
--dataset_name "cartoon" \
--real_images_path "dataset/testdata/cartoon/image" \
--reference_images_path "dataset/testdata/cartoon/reference" \
--generated_images_path_ft "results/cartoon/finetuned_model/results" \
--generated_images_path_p "results/cartoon/pretrained_model/results"

python fine_tuning_test_metric.py \
--dataset_name "handmake" \
--real_images_path "dataset/testdata/handmake/image" \
--reference_images_path "dataset/testdata/handmake/reference" \
--generated_images_path_ft "results/handmake/finetuned_model/results" \
--generated_images_path_p "results/handmake/pretrained_model/results"

python fine_tuning_test_metric.py \
--dataset_name "painting" \
--real_images_path "dataset/testdata/painting/image" \
--reference_images_path "dataset/testdata/painting/reference" \
--generated_images_path_ft "results/painting/finetuned_model/results" \
--generated_images_path_p "results/painting/pretrained_model/results"

python fine_tuning_test_metric.py \
--dataset_name "sketch" \
--real_images_path "dataset/testdata/sketch/image" \
--reference_images_path "dataset/testdata/sketch/reference" \
--generated_images_path_ft "results/sketch/finetuned_model/results" \
--generated_images_path_p "results/sketch/pretrained_model/results"

python fine_tuning_test_metric.py \
--dataset_name "tattoo" \
--real_images_path "dataset/testdata/tattoo/image" \
--reference_images_path "dataset/testdata/tattoo/reference" \
--generated_images_path_ft "results/tattoo/finetuned_model/results" \
--generated_images_path_p "results/tattoo/pretrained_model/results"

python fine_tuning_test_metric.py \
--dataset_name "weather" \
--real_images_path "dataset/testdata/weather/image" \
--reference_images_path "dataset/testdata/weather/reference" \
--generated_images_path_ft "results/weather/finetuned_model/results" \
--generated_images_path_p "results/weather/pretrained_model/results"
