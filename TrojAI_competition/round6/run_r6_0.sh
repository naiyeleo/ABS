# for i in 0000
# for i in {0000..0047}
# for i in 0006 0007 0008 0009 0010 0011 0018 0019 0020 0021 0022 0023 0030 0031 0032 0033 0034 0035 0042 0043 0044 0045 0046 0047
# for i in 0000 0001 0002 0003 0004 0005 0012 0013 0014 0015 0016 0017 0024 0025 0026 0027 0028 0029 0036 0037 0038 0039 0040 0041
for i in 0006
do
    dirname=id-0000$i
    echo $dirname
    time python ./abs_get_set.py --examples_dirpath=/mnt/32A6C453A6C418ED/trojai-round6-v2-dataset/models/$dirname/clean_example_data/ --model_filepath=/mnt/32A6C453A6C418ED/trojai-round6-v2-dataset/models/$dirname/model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch_2locs3_fast3/
done
 
