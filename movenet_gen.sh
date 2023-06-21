cd movenet

/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec1_crop.mp4 --model_name movenet_thunder
/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec2_crop.mp4 --model_name movenet_thunder
/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec3_crop.mp4 --model_name movenet_thunder
/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec4_crop.mp4 --model_name movenet_thunder

/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec1_crop.mp4 --model_name movenet_lightning
/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec2_crop.mp4 --model_name movenet_lightning
/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec3_crop.mp4 --model_name movenet_lightning
/usr/local/bin/python3.10 movenet_kp_gen.py --video_file ../../mcl_experiments/mcl_data/MCL_4_14/videos/rec4_crop.mp4 --model_name movenet_lightning

cd ..