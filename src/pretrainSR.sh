python Train.py --name s2_fsrcnn --model sr --sr-net fsrcnn --split 1 --dataset mnist --scale 2 --learning-rate 0.001 --batch-size 64
python Train.py --name s4_fsrcnn --model sr --sr-net fsrcnn --split 1 --dataset mnist --scale 4 --learning-rate 0.001 --batch-size 64  
python Train.py --name s2_wdsr --model sr --sr-net wdsr --split 1 --num-res-blocks 15 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.001 --batch-size 64
python Train.py --name s4_wdsr --model sr --sr-net wdsr --split 1 --num-res-blocks 15 --dataset svhn --num-images 200000 --scale 4 --learning-rate 0.001 --batch-size 64  
