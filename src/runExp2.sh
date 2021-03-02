
python Train.py --name s2_fsrcnn_sr2_wsr --pretrained-model s2_fsrcnn --model sr2 --sr-net fsrcnn --split 1 --dataset mnist --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.9 --weight-cl 0.1
python Train.py --name s4_fsrcnn_sr2_wsr --pretrained-model s4_fsrcnn --model sr2 --sr-net fsrcnn --split 1 --dataset mnist --scale 4 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.9 --weight-cl 0.1  
python Train.py --name s2_fsrcnn_sr2_wcl --pretrained-model s2_fsrcnn --model sr2 --sr-net fsrcnn --split 1 --dataset mnist --scale 2 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9
python Train.py --name s4_fsrcnn_sr2_wcl --pretrained-model s4_fsrcnn --model sr2 --sr-net fsrcnn --split 1 --dataset mnist --scale 4 --learning-rate 0.00001 --batch-size 64 --weight-sr 0.1 --weight-cl 0.9  

