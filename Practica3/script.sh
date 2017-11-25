echo Realizando ejecuciones
for k in grumpy720p.jpg grumpy1080p.jpg grumpy4k.jpg
do
echo "Imagen" $k
echo "Imagen" $k >> results.txt
for i in 80 100 200 400 800 1000
do
echo $i hilos
echo $i hilos >> results.txt
for j in $(seq 0 15)
do
d1=$(date "+%s%N")/1000000
./blureffect $k 21 $i
d2=$(date "+%s%N")/1000000
echo $(($d2 - $d1)) >> results.txt
done
done
done

