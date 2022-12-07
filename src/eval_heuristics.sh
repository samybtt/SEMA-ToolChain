search_dir="/media/sambt/1341-CC90/final/"
for family in $(ls $search_dir)
do
    echo $family
    for file in $(ls $search_dir/$entry | head -n 3)
    do
        for method in 
        do 
            $(python ToolChainSCDG/ToolChainSCDG.py --method=$method $search_dir/$family/$file --familly=$family)
        done
        # echo $file
    done
done