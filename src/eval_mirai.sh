search_dir="/root/SEMA-ToolChain/src/databases/malware-linux/mirai/"
# MethodArray=("CSTOCH CDFS CBFS STOCH WSELECT DFS BFS")
MethodArray=("CSTOCH CDFS CBFS STOCH WSELECT")

file1="5adf25df621f5a2d55a5d277ff9eb4a160e8806e8484d7ea4aa447173acd6dd3.elf"
family="mirai"



for file in $(ls $search_dir | tail -n 4)
    do
    echo $file
    for method in ${MethodArray[*]}
    do 
        echo $method
        echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$file --familly=$family --exp_dir=output/eval_SCDG/$method/ --dir=output/eval_SCDG/$method/ --restart_prob=0.00001"
        $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$file --familly=$family --exp_dir=output/eval_SCDG/$method/ --dir=output/eval_SCDG/$method/ --restart_prob=0.00001)
    done
done

for method in ${MethodArray[*]}
do 
    echo $method
    echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$file1 --familly=$family --exp_dir=output/eval_SCDG/$method/ --dir=output/eval_SCDG/$method/ --restart_prob=0.00001"
    $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$file1 --familly=$family --exp_dir=output/eval_SCDG/$method/ --dir=output/eval_SCDG/$method/ --restart_prob=0.00001)
done