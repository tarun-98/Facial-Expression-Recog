[train][201104_012643]python main.py --mode train --data_root datasets/CKPlus --train_csv train_ids_0.csv --print_losses_freq 4 --use_data_augment --visdom_env res_cls_ckp_0 --niter 100 --niter_decay 100 --gpu_ids 0 --model res_cls --solver res_cls --lambda_resface 0.1 --batch_size 16 --backend_pretrain --load_model_dir ckpts/CKPlus/res_baseline/fold_0/201101_084254 --load_epoch 300
[ test][201104_151851]python main.py --mode test --data_root datasets/CKPlus --test_csv test_ids_0.csv --gpu_ids 0 --model res_cls --solver res_cls --batch_size 4 --load_model_dir ckpts/CKPlus/res_cls/fold_0/201104_012643 --load_epoch 200
[ test][201104_152911]python main.py --mode test --data_root datasets/CKPlus --test_csv test_ids_1.csv --gpu_ids 0 --model res_cls --solver res_cls --batch_size 4 --load_model_dir ckpts/CKPlus/res_cls/fold_0/201104_012643 --load_epoch 200
[ test][201104_153136]python main.py --mode test --data_root datasets/CKPlus --test_csv test_ids_2.csv --gpu_ids 0 --model res_cls --solver res_cls --batch_size 4 --load_model_dir ckpts/CKPlus/res_cls/fold_0/201104_012643 --load_epoch 200
[ test][201104_153639]python main.py --mode test --data_root datasets/CKPlus --test_csv test_ids_3.csv --gpu_ids 0 --model res_cls --solver res_cls --batch_size 4 --load_model_dir ckpts/CKPlus/res_cls/fold_0/201104_012643 --load_epoch 200
