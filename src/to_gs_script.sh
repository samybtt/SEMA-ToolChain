search_dir="/root/final/"

MethodArray=("WSELECT WSELECT2 WSELECTSET2 CDFS CBFS CSTOCH CSTOCH2 CSTOCHSET2 STOCH")
FamArray=("bancteian ircbot sillyp2p sytro simbot FeakerStealer sfone lamer RedLineStealer RemcosRAT Sodinokibi delf nitol gandcrab wabot")

for method in ${MethodArray[*]}
do
    # $(mkdir databases/examples_samy/test_data/$method/)
    for family in ${FamArray[*]}
    do  
        # $(mkdir databases/examples_samy/test_data/$method/$family/)
        Path="output/eval_SCDG_n/$method/0/$family/"
        # Files=$(ls $Path | grep SCDG_ | grep .json)
        Files=$(ls databases/examples_samy/train_data/$method/$family/ | head -n 15 | tail -n 5)
        for file in ${Files[*]}
        do
            $(mv databases/examples_samy/train_data/$method/$family/$file databases/examples_samy/test_data/$method/$family/ )
            # Name=$(basename $file .json)
            # echo $Name
            # # echo $Path$file
            # $(python json_to_gs.py --outfile databases/examples_samy/train_data/$method/$family/$Name.gs $Path$file)
        done
    done
done





# for method in WSELECT WSELECT2 WSELECTSET2 CDFS CBFS CSTOCH CSTOCH2 CSTOCHSET2 STOCH; do     echo "Method $method:";     for family in bancteian ircbot sillyp2p sytro simbot FeakerStealer sfone lamer RedLineStealer RemcosRAT Sodinokibi delf nitol gandcrab wabot; do         num_files=$(ls "${method}/${family}/SCDG_"*.gs | wc -l);         echo "    Family ${family%/} has ${num_files} GS files.";     done; done