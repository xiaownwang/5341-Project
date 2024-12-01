python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/cartoon/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/cartoon/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/cartoon'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/handmake/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/handmake/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/handmake'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/painting/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/painting/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/painting'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/sketch/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/sketch/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/sketch'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/tattoo/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/tattoo/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/tattoo'

python -u fine_tuning.py \
--logdir models/Paint-by-Example \
--base configs/v1_fine_tuning.yaml \
--scale_lr False \
--pretrained_model checkpoints/model.ckpt \
--annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/weather/annotations.json' \
--coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/weather/images' \
--ckpt_save='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/checkpoints/weather'
