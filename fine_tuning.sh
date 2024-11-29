python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/cartoon/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/cartoon/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/cartoon'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/handmake/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/handmake/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/handmake'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/painting/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/painting/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/painting'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/sketch/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/sketch/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/sketch'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/tattoo/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/tattoo/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/tattoo'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/weather/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/weather/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/weather'

