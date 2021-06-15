# for i in 0014 0016
for i in 0000 0001 0002 0003 0014 0016 0010 0013 0006 0008
# for i in 0006 0008
# for i in 0000
do
    dirname=id-0000$i
    echo $dirname
    time python ./sample_normal_embs_abs_submit.py --examples_dirpath=/data/share/trojai/trojai-round5-v2-dataset/models/$dirname/clean_example_data/ --model_filepath=/data/share/trojai/trojai-round5-v2-dataset/models/$dirname/model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/
done
 
