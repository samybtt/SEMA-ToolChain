search_dir="/root/final/"
# MethodArray=("CSTOCH CDFS CBFS STOCH WSELECT DFS BFS")
# RandMethodArray=("CSTOCH CSTOCH2 CSTOCHset CSTOCHset2 WSELECT WSELECT2 WSELECTset WSELECTset2 STOCH")
RandMethodArray=("CSTOCH CSTOCH2 CSTOCHSET2 STOCH")
MethodArray=("WSELECT WSELECT2 WSELECTSET2 CDFS CBFS")
# FamArray=("bancteian ircbot sillyp2p sytro simbot FeakerStealer sfone lamer RedLineStealer RemcosRAT Sodinokibi delf nitol gandcrab wabot")
# FamArray=("lamer RedLineStealer Sodinokibi sillyp2p")
FamArray=("RedLineStealer FeakerStealer sfone lamer delf RemcosRAT Sodinokibi wabot gandcrab simbot sytro sillyp2p ircbot nitol bancteian")

j="0"
echo $j
for family in ${FamArray[*]}
do
    echo family
    for file in $(ls $search_dir/$family | tail -n 30 | head -n 10)
    do
        echo $file
        for method in ${RandMethodArray[*]}
        do
            echo $method
            echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/"
            $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/)
        done
        for method in ${MethodArray[*]}
        do 
            echo $method
            echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/"
            $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$j/ --dir=output/eval_SCDG_n/$method/$j/)
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
                echo "python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$i/ --dir=output/eval_SCDG_n/$method/$i/"
                $(python3 ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family --exp_dir=output/eval_SCDG_n/$method/$i/ --dir=output/eval_SCDG_n/$method/$i/)
            done
        done
    done
done