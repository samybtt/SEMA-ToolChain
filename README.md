# :skull_and_crossbones: SEMA :skull_and_crossbones: - ToolChain using Symbolic Execution for Malware Analysis. 

```
  ██████ ▓█████  ███▄ ▄███▓ ▄▄▄      
▒██    ▒ ▓█   ▀ ▓██▒▀█▀ ██▒▒████▄    
░ ▓██▄   ▒███   ▓██    ▓██░▒██  ▀█▄  
  ▒   ██▒▒▓█  ▄ ▒██    ▒██ ░██▄▄▄▄██ 
▒██████▒▒░▒████▒▒██▒   ░██▒ ▓█   ▓██▒
▒ ▒▓▒ ▒ ░░░ ▒░ ░░ ▒░   ░  ░ ▒▒   ▓▒█░
░ ░▒  ░ ░ ░ ░  ░░  ░      ░  ▒   ▒▒ ░
░  ░  ░     ░   ░      ░     ░   ▒   
      ░     ░  ░       ░         ░  ░
                                     
```
                                                                                                               
                                   
# :books:  Documentation

1. [ Architecture ](#arch)
    1. [ Toolchain architecture ](#arch_std)

2. [ Installation ](#install)

3. [ SEMA ](#tc)
    1. [ `SemaSCDG` ](#tcscdg)

4. [Quick Start Demos](#)
    1. [ `Extract SCDGs from binaries` ](https://github.com/csvl/SEMA-ToolChain/blob/production/Tutorial/Notebook/SEMA-SCDG%20Demo.ipynb)

5. [ Credentials ](#credit)

:page_with_curl: Architecture
====
<a name="arch"></a>

### Toolchain architecture
<a name="arch_std"></a>


##### Main depencies: 

    * Python 3.8 (angr)

    * KVM/QEMU

    * Celery

##### Interesting links

* https://angr.io/

* https://bazaar.abuse.ch/

:page_with_curl: Installation
====
<a name="install"></a>

Tested on Ubuntu 18 LTS. Checkout Makefile and install.sh for more details.

**Recommanded installation:**

```bash
git clone https://github.com/Manon-Oreins/SEMA-ToolChain.git;
# Full installation (ubuntu)
make build-toolchain;
```

## Installation details (optional)

#### Pip

To run this SCDG extractor you first need to install pip.

##### Debian (and Debian-based)
To install pip on debian-based systems:
```bash
sudo apt update;
sudo apt-get install python3-pip xterm;
```

##### Arch (and Arch-based)
To install pip on arch-based systems:
```bash
sudo pacman -Sy python-pip xterm;
```

#### Python virtual environment

For `angr`, it is recommended to use the python virtual environment. 

```bash
python3 -m venv penv;
```

This create a virtual envirnment called `penv`. 
Then, you can run your virtual environment with:

```bash
source penv/bin/activate;
```

##### For extracting test database

```bash
cd databases/Binaries; bash extract_deploy_db.sh
```

##### For code cleaning

For dev (code cleaning):
```bash
cd databases/Binaries; bash compress_db.sh 
#To zip back the test database
make clean-scdg-empty-directory
#To remove all directory created by the docker files that are empty

```

##### Dependencies

docker buildx and docker compose -> see https://docs.docker.com/engine/install/ubuntu/

:page_with_curl: `SEMA - ToolChain`
====
<a name="tc"></a>

Our toolchain is represented in the next figure  and works as follow. A collection of labelled binaries of different malwares families is collected and used as the input of the toolchain. **Angr**, a framework for symbolic execution, is used to execute symbolically binaries and extract execution traces. For this purpose, different heuristics have been developped to optimize symbolic execution. Several execution traces (i.e : API calls used and their arguments) corresponding to one binary are extracted with Angr and gather together thanks to several graph heuristics to construct a SCDG. These resulting SCDGs are then used as input to graph mining to extract common graph between SCDG of the same family and create a signature. Finally when a new sample has to be classified, its SCDG is build and compared with SCDG of known families (thanks to a simple similarity metric).


### How to use ?

First launch the containers : 
```bash
make run-toolchain
```
Wait for the containers to be up

Then visit 127.0.0.1:5000 on your browser


:page_with_curl: System Call Dependency Graphs extractor (`SemaSCDG`)
====
<a name="tcscdg"></a>

This repository contains a first version of a SCDG extractor.
During symbolic analysis of a binary, all system calls and their arguments found are recorded. After some stop conditions for symbolic analysis, a graph is build as follow : Nodes are systems Calls recorded, edges show that some arguments are shared between calls.

### How to use ?
First run the SCDG container:
```bash
make run-scdg-service
```

Inside the container just run  :
```bash
python3 SemaSCDG.py config.ini
```
Or if you want to use pypy3:
```bash
pypy3 SemaSCDG.py config.ini
```

The parameters are put in a configuration file : "config.ini"
Feel free to modify it or create new configuration files to run different experiments. 
To restore the default values of 'config.ini' do :
```bash
python3 restore_defaults.py
```
The default parameters are stored in the file "default_config.ini"

**The binary path has to be a relative path to a binary beeing into the *database* directory**

### Parameters description
SCDG module arguments

```
expl_method:
  DFS                 Depth First Search
  BFS                 Breadth First Search
  CDFS                TODO
  CBFS                TODO (default)
  DBFS                TODO
  SDFS                TODO
  SCDFS               TODO

graph_output:
  gs                  .GS format
  json                .JSON format
  EMPTY               if left empty then build on all available format

packing_type:
  symbion             Concolic unpacking method (linux | windows [in progress])
  unipacker           Emulation unpacking method (windows only)

SCDG exploration techniques parameters:
  jump_it              Number of iteration allowed for a symbolic loop (default : 3)
  max_in_pause_stach   Number of states allowed in pause stash (default : 200)
  max_step             Maximum number of steps allowed for a state (default : 50 000)
  max_end_state        Number of deadended state required to stop (default : 600)
  max_simul_state      Number of simultaneous states we explore with simulation manager (default : 5)

Binary parameters:
  n_args                  Number of symbolic arguments given to the binary (default : 0)
  loop_counter_concrete   TODO (default : 10240)
  count_block_enable      Enable the count of visited blocks and instructions
  sim_file                Create SimFile
  entry_addr              Entry address of the binary

SCDG creation parameter:
  min_size             Minimum size required for a trace to be used in SCDG (default : 3)
  disjoint_union       Do we merge traces or use disjoint union ? (default : merge)
  not_comp_args        Do we compare arguments to add new nodes when building graph ? (default : comparison enabled)
  three_edges          Do we use the three-edges strategy ? (default : False)
  not_ignore_zero      Do we ignore zero when building graph ? (default : Discard zero)
  keep_inter_SCDG      Keep intermediate SCDG in file (default : False)
  eval_time            TODO

Global parameter:
  concrete_target_is_local      Use a local GDB server instead of using cuckoo (default : False)
  print_syscall        print the syscall found
  print_address        print the address
  csv_file             Name of the csv to save the experiment data
  plugin_enable        enable the plugins set to true in the config.ini file
  approximate          Symbolic approximation
  is_packed            Is the binary packed ? (default : False)
  timeout              Timeout in seconds before ending extraction (default : 600)
  string_resolve       Do we try to resolv references of string (default : False)
  memory_limit         Skip binary experiment when memory > 90% (default : False)
  log_level            Level of log, can be INFO, DEBUG, WARNING, ERROR (default : INFO) 
  family               Family of the malware (default : Unknown)
  exp_dir              Name of the directory to save SCDG extracted (default : Default)
  binary_path          Path to the binary or directory
  fast_main            Jump directly into the main function 
  
Plugins:
  plugin_env_var          Enable the env_var plugin 
  plugin_locale_info      Enable the locale_info plugin
  plugin_resources        Enable the resources plugin
  plugin_widechar         Enable the widechar plugin
  plugin_registery        Enable the registery plugin
  plugin_atom             Enable the atom plugin
  plugin_thread           Enable the thread plugin
  plugin_track_command    Enable the track_command plugin
  plugin_ioc_report       Enable the ioc_report plugin
  plugin_hooks            Enable the hooks plugin
```

To know the details of the angr options see [Angr documentation](https://docs.angr.io/en/latest/appendix/options.html)

Program will output a graph in `.gs` format that could be exploited by `gspan`.

You also have a script `MergeGspan.py` which could merge all `.gs` from a directory into only one file.

Password for Examples archive is "infected". Warning : it contains real samples of malwares.


## Managing your runs

If you want to remove all the runs you have made :
```bash
make clean-scdg-runs
```

If you want to save some runs into the saved_runs file:
```bash
make save-scdg-runs                   #If you want to save all runs
make save-scdg-runs ARGS=DIR_NAME     #If you want to save only a specific run
```

If you want to erase all saved runs :
```bash
make clean-scdg-saved-runs
```

:page_with_curl: Model & Classification extractor (`SemaClassifier`)
====
<a name="tcc"></a>

When a new sample has to be evaluated, its SCDG is first build as described previously. Then, `gspan` is applied to extract the biggest common subgraph and a similarity score is evaluated to decide if the graph is considered as part of the family or not.

The similarity score `S` between graph `G'` and `G''` is computed as follow:

Since `G''` is a subgraph of `G'`, this is calculating how much `G'` appears in `G''`.

Another classifier we use is the Support Vector Machine (`SVM`) with INRIA graph kernel or the Weisfeiler-Lehman extension graph kernel.

### How to use ?

Just run the script : 
```bash
python3 SemaClassifier.py FOLDER/FILE

usage: update_readme_usage.py [-h] [--threshold THRESHOLD] [--biggest_subgraph BIGGEST_SUBGRAPH] [--support SUPPORT] [--ctimeout CTIMEOUT] [--epoch EPOCH] [--sepoch SEPOCH]
                              [--data_scale DATA_SCALE] [--vector_size VECTOR_SIZE] [--batch_size BATCH_SIZE] (--classification | --detection) (--wl | --inria | --dl | --gspan)
                              [--bancteian] [--delf] [--FeakerStealer] [--gandcrab] [--ircbot] [--lamer] [--nitol] [--RedLineStealer] [--sfone] [--sillyp2p] [--simbot]
                              [--Sodinokibi] [--sytro] [--upatre] [--wabot] [--RemcosRAT] [--verbose_classifier] [--train] [--nthread NTHREAD]
                              binaries

Classification module arguments

optional arguments:
  -h, --help            show this help message and exit
  --classification      By malware family
  --detection           Cleanware vs Malware
  --wl                  TODO
  --inria               TODO
  --dl                  TODO
  --gspan               TODOe

Global classifiers parameters:
  --threshold THRESHOLD
                        Threshold used for the classifier [0..1] (default : 0.45)

Gspan options:
  --biggest_subgraph BIGGEST_SUBGRAPH
                        Biggest subgraph consider for Gspan (default: 5)
  --support SUPPORT     Support used for the gpsan classifier [0..1] (default : 0.75)
  --ctimeout CTIMEOUT   Timeout for gspan classifier (default : 3sec)

Deep Learning options:
  --epoch EPOCH         Only for deep learning model: number of epoch (default: 5) Always 1 for FL model
  --sepoch SEPOCH       Only for deep learning model: starting epoch (default: 1)
  --data_scale DATA_SCALE
                        Only for deep learning model: data scale value (default: 0.9)
  --vector_size VECTOR_SIZE
                        Only for deep learning model: Size of the vector used (default: 4)
  --batch_size BATCH_SIZE
                        Only for deep learning model: Batch size for the model (default: 1)

Malware familly:
  --bancteian
  --delf
  --FeakerStealer
  --gandcrab
  --ircbot
  --lamer
  --nitol
  --RedLineStealer
  --sfone
  --sillyp2p
  --simbot
  --Sodinokibi
  --sytro
  --upatre
  --wabot
  --RemcosRAT

Global parameter:
  --verbose_classifier  Verbose output during train/classification (default : False)
  --train               Launch training process, else classify/detect new sample with previously computed model
  --nthread NTHREAD     Number of thread used (default: max)
  binaries              Name of the folder containing binary'signatures to analyze (Default: output/save-SCDG/, only that for ToolChain)

```

#### Example

This will train models for input dataset

```bash
python3 SemaClassifier/SemaClassifier.py --train output/save-SCDG/
```

This will classify input dataset based on previously computed models

```bash
python3 SemaClassifier/SemaClassifier.py output/test-set/
```

## Shut down

To leave the toolchain just press Ctrl+C then use

```bash
make stop-all-containers
```

To stop all docker containers.

If you want to remove all images :

```bash
docker rmi sema-web-app
docker rmi sema-scdg
docker rmi sema-classifier
```

:page_with_curl: Credentials
====
<a name="credit"></a>

Main authors of the projects:

* **Charles-Henry Bertrand Van Ouytsel** (UCLouvain)

* **Christophe Crochet** (UCLouvain)

* **Khanh Huu The Dam** (UCLouvain)

* **Oreins Manon** (UCLouvain)

Under the supervision and with the support of **Fabrizio Biondi** (Avast) 

Under the supervision and with the support of our professor **Axel Legay** (UCLouvain) (:heart:)
