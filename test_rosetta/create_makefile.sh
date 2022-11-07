num_total_executions=1000
output_name=frag
n_gpu=0
command="python low_res_fold.py $output_name"
mkdir -p $output_name
mkdir -p $output_name/logs
echo -e "all:\tjob1\t\\" > Makefile
for i in `seq 2 $num_total_executions`
do
echo -e "\tjob$i\t\\" >> Makefile
done
echo " " >> Makefile
for i in `seq 1 $num_total_executions`
do
echo "job$i:" >> Makefile
if [ $n_gpu == 0 ]; then
device=cpu
else
cuda_idx=$((i%$n_gpu))
device="cuda:$cuda_idx"
fi
echo -e "\t$command $i $device > $output_name/logs/$i.out" >> Makefile
done
