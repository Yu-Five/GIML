

python ./main_noise_av.py --ckpt_path ./results/cramed/ --dataset CREMAD --gpu_ids 0 --num_frame 1 --pe 1  --beta 1e-5 --gamma 4.0 --batch_size 64

python ./main_noise_av.py --ckpt_path ./results/ks/ --dataset  KineticSound --gpu_ids 0 --num_frame 3 --pe 1  --beta 1e-5 --gamma 4.0 --batch_size 64

python ./main_noise_vt.py --ckpt_path ./results/mvsa/ --dataset MVSA_Single --gpu_ids 0 --pe 1  --beta 1e-5 --gamma 4.0 --num_frame 1 --batch_size 32 


python ./main_noise_avt.py --ckpt_path ./results/mosi/ --dataset MOSI --gpu_ids 0 --num_frame 5 --pe 1  --beta 1e-5 --gamma 4.0 --batch_size 32


python ./main_noise_rgbdepth.py --ckpt_path ./results/nvgesture/ --dataset NVGesture --gpu_ids 0,1 --pe 1  --beta 1e-5 --gamma 5.0 --num_frame 24 --batch_size 32 














