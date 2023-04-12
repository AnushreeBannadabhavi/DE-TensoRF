# Baselines
python train.py --config configs/lego_2.txt 
python train.py --config configs/lego_3.txt 
python train.py --config configs/lego_5.txt 
python train.py --config configs/lego_10.txt 

# With symmetry
python train.py --config configs/lego_sym_2.txt
python train.py --config configs/lego_sym_3.txt
python train.py --config configs/lego_sym_5.txt
python train.py --config configs/lego_sym_10.txt

# With cond
python train.py --config configs/lego_cond_2.txt
python train.py --config configs/lego_cond_3.txt
python train.py --config configs/lego_cond_5.txt
python train.py --config configs/lego_cond_10.txt

# With symmetry and cond
python train.py --config configs/lego_sym_cond_2.txt
python train.py --config configs/lego_sym_cond_3.txt
python train.py --config configs/lego_sym_cond_5.txt
python train.py --config configs/lego_sym_cond_10.txt

# With symmetry and cond and sem loss
python train.py --config configs/lego_sym_cond_sem_2.txt
python train.py --config configs/lego_sym_cond_sem_3.txt
python train.py --config configs/lego_sym_cond_sem_5.txt
python train.py --config configs/lego_sym_cond_sem_10.txt

# With sem loss
# python train.py --config configs/lego_sem_2.txt
# python train.py --config configs/lego_sem_3.txt
# python train.py --config configs/lego_sem_5.txt
# python train.py --config configs/lego_sem_10.txt


