[train][201101_084254]python main.py --mode train --data_root datasets/CKPlus --train_csv train_ids_0.csv --print_losses_freq 4 --use_data_augment --visdom_env res_baseline_ckp_0 --niter 150 --niter_decay 150 --gpu_ids 0 --model res_baseline --solver resface --img_nc 1 --batch_size 32