num_total_executions=1000
output_name=frag+int2cart
command="python low_res_fold.py $output_name"
mkdir -p $output_name
mkdir -p $output_name/logs
echo "all:\tjob1\t\\" > Makefile
for i in `seq 2 $num_total_executions`
do
echo "\tjob$i\t\\" >> Makefile
done
echo " " >> Makefile
for i in `seq 1 $num_total_executions`
do
echo "job$i:" >> Makefile
cuda_idx=$((i%4))
echo "\t$command $i cuda:$cuda_idx > $output_name/logs/$i.out" >> Makefile
done