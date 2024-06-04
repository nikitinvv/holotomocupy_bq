# energy = float(sys.argv[1])  # [keV] xray energy
# z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# # z1p = 270e-3# positions of the probe and code for reconstruction
# ill_feature_size = float(sys.argv[3])
# use_prb = sys.argv[4]=='True'
# use_code = sys.argv[5]=='True'
# ndist = int(sys.argv[6])
# smooth = int(sys.argv[7])
# flg_show = False

python data_modeling.py 25 8e-3 1e-6 True True 1 0 
python data_modeling.py 25 8e-3 1e-6 True True 1 5 
python data_modeling.py 25 8e-3 1e-6 True True 1 10 
python data_modeling.py 25 8e-3 1e-6 True True 1 20 
python data_modeling.py 25 8e-3 1e-6 True True 1 50 
python data_modeling.py 25 8e-3 1e-6 True True 1 100 
python data_modeling.py 25 8e-3 1e-6 True True 1 300 


python data_modeling.py 25 8e-3 1e-6 False True 1 0 
python data_modeling.py 25 8e-3 1e-6 False True 1 5 
python data_modeling.py 25 8e-3 1e-6 False True 1 10 
python data_modeling.py 25 8e-3 1e-6 False True 1 20 
python data_modeling.py 25 8e-3 1e-6 False True 1 50 
python data_modeling.py 25 8e-3 1e-6 False True 1 100 
python data_modeling.py 25 8e-3 1e-6 False True 1 300 



energy = float(sys.argv[1])  # [keV] xray energy
z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# z1p = 270e-3# positions of the probe and code for reconstruction
ill_feature_size = float(sys.argv[3])
use_prb = sys.argv[4]=='True'
use_code = sys.argv[5]=='True'
ndist = int(sys.argv[6])
niter = int(sys.argv[7])
step = int(sys.argv[8])
smooth = int(sys.argv[9])
flg_show = False

tomo1
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 25 8e-3 1e-6 True True 1 100000 256 100 >True_100 2>True_100 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 25 8e-3 1e-6 False True 1 100000 256 100 >False_100 2>False_100 &

tomo4
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 25 8e-3 1e-6 False True 1 100000 256 0 >False_0 2>False_0 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 25 8e-3 1e-6 True True 1 100000 256 0 >True_0 2>True_0 &
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 25 8e-3 1e-6 False True 1 100000 256 10 >False_10 2>False_10 &
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 25 8e-3 1e-6 True True 1 100000 256 10 >True_10 2>True_10 &





python data_modeling.py 33.35 12e-3 1e-6 False True 1 10 
python data_modeling.py 33.35 15e-3 1e-6 False True 1 10 



energy = float(sys.argv[1])  # [keV] xray energy
z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# z1p = 270e-3# positions of the probe and code for reconstruction
ill_feature_size = float(sys.argv[3])
use_prb = sys.argv[4]=='True'
use_code = sys.argv[5]=='True'
ndist = int(sys.argv[6])
smooth = int(sys.argv[7])
code_thickness = float(sys.argv[8])


python data_modeling.py 33.35 10e-3 1e-6 False True 2 2 2e-6
python data_modeling.py 33.35 8e-3 1e-6 False True 2 2 2e-6
python data_modeling.py 33.35 5e-3 1e-6 False True 2 2 2e-6
python data_modeling.py 33.35 12e-3 1e-6 False True 2 2 2e-6



energy = float(sys.argv[1])  # [keV] xray energy
z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# z1p = 270e-3# positions of the probe and code for reconstruction
ill_feature_size = float(sys.argv[3])
use_prb = sys.argv[4]=='True'
use_code = sys.argv[5]=='True'
ndist = int(sys.argv[6])
niter = int(sys.argv[7])
step = int(sys.argv[8])
smooth = int(sys.argv[9])
code_thickness = float(sys.argv[10])


CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 8e-3 1e-6 False True 2 10000 64 2 2e-6 >False_8 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 10e-3 1e-6 False True 2 10000 64 2 2e-6 >False_10 &

CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 12e-3 1e-6 False True 2 10000 64 2 2e-6 >False_12 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 5e-3 1e-6 False True 2 10000 64 2 2e-6 >False_5 &

CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 15e-3 1e-6 False True 2 10000 64 2 2e-6 >False_15 &

CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 10000 64 2 2e-6 >False_20 &


energy = float(sys.argv[1])  # [keV] xray energy
z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# z1p = 270e-3# positions of the probe and code for reconstruction
ill_feature_size = float(sys.argv[3])
use_prb = sys.argv[4]=='True'
use_code = sys.argv[5]=='True'
ndist = int(sys.argv[6])
niter = int(sys.argv[7])
step = int(sys.argv[8])
smooth = int(sys.argv[9])
code_thickness = float(sys.argv[10])
flg_show = False

# energy =33.35  # [keV] xray energy
# z1p = 8e-3# positions of the probe and code for reconstruction
# # z1p = 270e-3# positions of the probe and code for reconstruction
# ill_feature_size = 1e-6
# code_thickness = 1e-6
# use_prb = False
# use_code = False
# ndist = 1
# smooth = 3
# niter = 2049
# step = 32
# flg_show = True

#tomo5
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 8e-3 1e-6 False True 1 2049 64 2 2e-6 >False_81 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 12e-3 1e-6 False True 1 2049 64 2 2e-6 >False_121 &

#tomo4
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 8e-3 1e-6 False True 1 2049 64 2 1e-6 >False_82 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 12e-3 1e-6 False True 1 2049 64 2 1e-6 >False_122 &


CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 10e-3 1e-6 False True 2 2049 64 2 1e-6 >False_1022 &





energy = float(sys.argv[1])  # [keV] xray energy
z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# z1p = 270e-3# positions of the probe and code for reconstruction
ill_feature_size = float(sys.argv[3])
use_prb = sys.argv[4]=='True'
use_code = sys.argv[5]=='True'
ndist = int(sys.argv[6])
smooth = int(sys.argv[7])
code_thickness= float(sys.argv[8])
vc = float(sys.argv[9])
flg_show=False

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 5 2e-6 0.7 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 5 2e-6 1 &\
bash'"


ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 2 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 2 5 2e-6 0.7 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 2 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 2 5 2e-6 1 &\
bash'"


ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 5 2e-6 0.7 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 5 2e-6 1 &\
bash'"


ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 2e-6 0.7 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 2e-6 1 &\
bash'"




ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 2049 64 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 2049 64 5 2e-6 0.7 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 2049 64 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 2049 64 5 2e-6 1 &\
bash'"

ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 2e-6 0.7 &\
bash'"

ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 2e-6 1 &\
bash'"




ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 3 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 3 5 2e-6 0.7 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 3 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 3 5 2e-6 1 &\
bash'"



ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 3 2049 64 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 3 2049 64 5 2e-6 0.7 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 3 2049 64 5 2e-6 0.8 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 3 2049 64 5 2e-6 1 &\
bash'"



ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 2e-6 0.6 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 2e-6 0.6 &\
bash'"

ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 5 2e-6 0.5 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 5 2e-6 0.5 &\
bash'"


ssh -t tomo@tomo1 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 2e-6 0.5 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 2e-6 0.5 &\
bash'"
#############################################################


ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 5 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 2 5 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 3 5 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 5 2e-6 0.6 >False1 &\
bash'"

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 2049 64 5 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 3 2049 64 5 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 2e-6 0.6 >False1 &\
bash'"

##########

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 25 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 2 25 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 3 25 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 25 2e-6 0.6 >False1 &\
bash'"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 25 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 2049 64 25 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 3 2049 64 25 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 25 2e-6 0.6 >False1 &\
bash'"



##########

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; \
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 1 5 1e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 2 5 1e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 20e-3 1e-6 False True 3 5 1e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 20e-3 1e-6 False False 1 5 1e-6 0.6 >False1 &\
bash'"

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 1 2049 64 5 1e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 2 2049 64 5 1e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 20e-3 1e-6 False True 3 2049 64 5 1e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 20e-3 1e-6 False False 1 2049 64 5 1e-6 0.6 >False1 &\
bash'"


##########

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 10e-3 1e-6 False True 1 5 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 10e-3 1e-6 False True 2 5 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 10e-3 1e-6 False True 3 5 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 10e-3 1e-6 False False 1 5 2e-6 0.6 >False1 &\
bash'"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 10e-3 1e-6 False True 1 2049 64 5 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 10e-3 1e-6 False True 2 2049 64 5 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 10e-3 1e-6 False True 3 2049 64 5 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 10e-3 1e-6 False False 1 2049 64 5 2e-6 0.6 >False1 &\
bash'"

##############
ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; \
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling.py 33.35 30e-3 1e-6 False True 1 5 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling.py 33.35 30e-3 1e-6 False True 2 5 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling.py 33.35 30e-3 1e-6 False True 3 5 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling.py 33.35 30e-3 1e-6 False False 1 5 2e-6 0.6 >False1 &\
bash'"

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 33.35 30e-3 1e-6 False True 1 2049 64 5 2e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 33.35 30e-3 1e-6 False True 2 2049 64 5 2e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 33.35 30e-3 1e-6 False True 3 2049 64 5 2e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 33.35 30e-3 1e-6 False False 1 2049 64 5 2e-6 0.6 >False1 &\
bash'"


#####





ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False True 1 5 1e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False True 2 5 1e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False True 3 5 1e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False False 1 5 1e-6 0.6 >False1 &\
bash'"

ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False True 1 2049 64 5 1e-6 0.6 >True1 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False True 2 2049 64 5 1e-6 0.6 >True2 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False True 3 2049 64 5 1e-6 0.6 >True3 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False False 1 2049 64 5 1e-6 0.6 >False1 &\
bash'"



ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False True 1 5 1e-6 1 >True12 &\
CUDA_VISIBLE_DEVICES=1 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False True 2 5 1e-6 1 >True22 &\
CUDA_VISIBLE_DEVICES=2 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False True 3 5 1e-6 1 >True33 &\
CUDA_VISIBLE_DEVICES=3 nohup python data_modeling_big.py 33.35 20e-3 1e-6 False False 1 5 1e-6 1 >False11 &\
bash'"

ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomo; cd vnikitin/holotomo-dev2/tests/3d_ald_syn_codes/; pkill -9 python;\
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False True 1 2049 64 5 1e-6 1 >True11 &\
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False True 2 2049 64 5 1e-6 1 >True22 &\
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False True 3 2049 64 5 1e-6 1 >True33 &\
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm_big.py 33.35 20e-3 1e-6 False False 1 2049 64 5 1e-6 1 >False11 &\
bash'"
