

python Train.py --name s2_wdsr_split1 --pretrained-model s2_wdsr --model sr2 --sr-net wdsr --split 1 --num-res-blocks 15 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s4_wdsr_split1 --pretrained-model s4_wdsr --model sr2 --sr-net wdsr --split 1 --num-res-blocks 15 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s2_wdsr_split2 --pretrained-model s2_wdsr --model sr2 --sr-net wdsr --split 2 --num-res-blocks 14 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s4_wdsr_split2 --pretrained-model s4_wdsr --model sr2 --sr-net wdsr --split 2 --num-res-blocks 14 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s2_wdsr_split3 --pretrained-model s2_wdsr --model sr2 --sr-net wdsr --split 3 --num-res-blocks 13 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s4_wdsr_split3 --pretrained-model s4_wdsr --model sr2 --sr-net wdsr --split 3 --num-res-blocks 13 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s2_wdsr_split4 --pretrained-model s2_wdsr --model sr2 --sr-net wdsr --split 4 --num-res-blocks 12 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s4_wdsr_split4 --pretrained-model s4_wdsr --model sr2 --sr-net wdsr --split 4 --num-res-blocks 12 --dataset svhn --num-images 200000 --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9


