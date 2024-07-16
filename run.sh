# 运行 LightningCLI 命令并设置 TensorBoardLogger 参数
# 显卡会影响结果，最优值在A100-sxm4-80gb
python main.py fit\
    --seed_everything 3407 \
    --trainer.max_epochs 15 \
    --model.bert_lr 1e-6 \
    --model.lr 5e-5 \
    --model.max_len 128 \
    --model.model_name early_dec_esdm \
    --data.num_workers=4 \
    --data.max_len 128 \
    --data.model_name early_dec_esdm \
    --trainer.logger TensorBoardLogger \
    --trainer.logger.init_args.save_dir ~/Logs \
    --trainer.logger.init_args.name EarlyDecEsdm \
    --trainer.logger.init_args.default_hp_metric false \
    --trainer.callbacks lightning.pytorch.callbacks.ModelCheckpoint \
    --trainer.callbacks.init_args.every_n_epochs 1 \
    --trainer.callbacks.init_args.save_top_k -1