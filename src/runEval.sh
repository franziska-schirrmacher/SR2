noiseLevel=(0.00001 0.00005 0.0001 0.0005 0.001)

echo Starting experiment!

for i in "${noiseLevel[@]}"; do

python Test.py --dataset svhn --scale 2 --noiseType gaussian --noiseLow $i  --noiseHigh $i

done
