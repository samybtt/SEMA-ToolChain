search_dir="/root/final/"
# MethodArray=("CSTOCH CDFS CBFS STOCH WSELECT DFS BFS")
# RandMethodArray=("CSTOCH CSTOCH2 CSTOCHset CSTOCHset2 WSELECT WSELECT2 WSELECTset WSELECTset2 STOCH")
RandMethodArray=("CSTOCH CSTOCH2 CSTOCHSET2 STOCH")
MethodArray=("WSELECT WSELECT2 WSELECTSET2 CDFS CBFS")
FamArray=("bancteian ircbot sillyp2p sytro simbot FeakerStealer sfone lamer RedLineStealer RemcosRAT Sodinokibi delf nitol gandcrab wabot")
# FamArray=("lamer RedLineStealer Sodinokibi sillyp2p")

j="0"
echo $j
for family in ${FamArray[*]}
do
    echo family
    for file in $(ls $search_dir/$family | head -n 20)
    do
        echo $file
        for method in ${RandMethodArray[*]}
        do
            echo $method
            echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/ --format_out=json"
            $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/ --format_out=json)
        done
        for method in ${MethodArray[*]}
        do 
            echo $method
            echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/ --format_out=json"
            $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/ --format_out=json)
        done
    done
done


for i in {1..10}
do
    echo $i
    for family in ${FamArray[*]}
    do
        echo family
        for file in $(ls $search_dir/$family | head -n 20)
        do 
            echo $file
            for method in ${RandMethodArray[*]}
            do
                echo $method
                echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$i/ --dir=output/eval_SCDG_n/$method/$i/ --format_out=json"
                $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$i/ --dir=output/eval_SCDG_n/$method/$i/ --format_out=json)
            done
        done
    done
done