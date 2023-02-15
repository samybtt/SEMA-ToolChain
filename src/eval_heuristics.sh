search_dir="/root/final/"
# MethodArray=("CSTOCH CDFS CBFS STOCH WSELECT DFS BFS")
MethodArray=("CSTOCH CDFS CBFS STOCH WSELECT")
# FamArray=("delf nitol bancteian gandcrab ircbot lamer RedLineStealer RemcosRAT sfone simbot Sodinokibi sytro sillyp2p wabot FeakerStealer")
FamArray=("lamer RedLineStealer Sodinokibi sillyp2p")

for family in ${FamArray[*]}
do
    echo $family
    for file in $(ls $search_dir/$family | head -n 4)
    do
        echo $file
        for method in ${MethodArray[*]}
        do 
            echo $method
            echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG/$method --restart_prob=0.00001"
            $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG/$method --restart_prob=0.00001)
        done
        # echo $file
    done
done