import logging
import os
import time
import dill

try:
    from classifier.GM.GSpanClassifier import GSpanClassifier
    from helper.ArgumentParserClassifier import ArgumentParserClassifier
    from classifier.SVM.SVMInriaClassifier import SVMInriaClassifier
    from classifier.SVM.SVMWLClassifier import SVMWLClassifier
    # from classifier.GNN.GraphSAGEClassifier import GraphSAGE
    from classifier.GNN.GNNTrainer import GNNTrainer
    from clogging.CustomFormatter import CustomFormatter
except:
    from .classifier.GM.GSpanClassifier import GSpanClassifier
    from .helper.ArgumentParserClassifier import ArgumentParserClassifier
    from .classifier.SVM.SVMInriaClassifier import SVMInriaClassifier
    from .classifier.SVM.SVMWLClassifier import SVMWLClassifier
    from .classifier.GNN.GNNTrainer import GNNTrainer
    from .clogging.CustomFormatter import CustomFormatter


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# TODO make usage of method in main file
class ToolChainClassifier:
    def __init__(self, classifier_name="wl",parse=True):
        self.classifier = None
        self.input_path = None
        self.mode = "classification"
        self.classifier_name = classifier_name
        self.start_time = time.time()
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(CustomFormatter())
        self.log = logging.getLogger("ToolChainClassifier")
        self.log.setLevel(logging.INFO)
        self.log.addHandler(ch)
        self.log.propagate = False    
        self.args = None  
        self.families = []      

    def save_model(self,object, path):
        with open(path, 'wb+') as output:
            dill.dump(object, output)

    def load_model(self,path):
        with open(path, 'rb') as inp:
            return dill.load(inp)

    def init_classifer(self,args,
                families=['bancteian','delf','FeakerStealer','gandcrab','ircbot','lamer','nitol','RedLineStealer','sfone','sillyp2p','simbot','Sodinokibi','sytro','upatre','wabot','RemcosRAT'],
                is_fl=False, from_saved_model=False):
        self.log.info(args)
        if not is_fl:
            threshold = args.threshold
            support = args.support
            ctimeout = args.ctimeout
            nthread = args.nthread
            biggest_subgraph = args.biggest_subgraph
            epoch = args.epoch
            shared_type = 1#args.smodel
            self.mode = args.mode
        else:
            threshold = args["threshold"]
            support = args["support"]
            ctimeout = args["ctimeout"]
            nthread = args["nthread"]
            biggest_subgraph = args["biggest_subgraph"]
            epoch = args["epoch"]
            shared_type = 1#args["smodel"]
        if not from_saved_model:
            if self.classifier_name == "gspan":
                self.classifier = GSpanClassifier(path=ROOT_DIR,threshold=threshold,support=support,timeout=ctimeout,thread=nthread,biggest_subgraphs=biggest_subgraph)
            elif self.classifier_name == "inria": 
                self.classifier = SVMInriaClassifier(path=ROOT_DIR,threshold=threshold,families=families)
            elif self.classifier_name == "wl": 
                # import pdb; pdb.set_trace()
                self.classifier = SVMWLClassifier(path=ROOT_DIR,threshold=threshold,families=families)
            elif self.classifier_name == "dl": # not working with pypy
                try:
                    from classifier.DL.DLTrainerClassifier import DLTrainerClassifier
                except:
                    from .classifier.DL.DLTrainerClassifier import DLTrainerClassifier
                self.classifier = DLTrainerClassifier(path=ROOT_DIR,epoch=epoch,shared_type=shared_type)
            elif self.classifier_name == "gin":
                self.classifier = GNNTrainer(path=ROOT_DIR, name="gin", threshold=threshold, families=families)
            elif self.classifier_name == "sage":
                self.classifier = GNNTrainer(path=ROOT_DIR, name="sage", threshold=threshold, families=families)
            elif self.classifier_name == "ginjk":
                self.classifier = GNNTrainer(path=ROOT_DIR, name="ginjk", threshold=threshold, families=families)
            else:
                self.log.info("Error: Unrecognize classifer (gspan|inria|wl|dl)")
                exit(-1)    
        else: # TODO improve
            if self.classifier_name == "gspan":
                self.classifier = self.load_model(ROOT_DIR + "/classifier/saved_model/gspan_model.pkl")
            elif self.classifier_name == "inria": 
                self.classifier = self.load_model(ROOT_DIR + "/classifier/saved_model/inria_model.pkl")
            elif self.classifier_name == "wl": 
                self.classifier = self.load_model(ROOT_DIR + "/classifier/saved_model/wl_model.pkl")
            elif self.classifier_name == "dl": # not working with pypy
                try:
                    from classifier.DL.DLTrainerClassifier import DLTrainerClassifier
                except:
                    from .classifier.DL.DLTrainerClassifier import DLTrainerClassifier
                self.classifier = self.load_model(ROOT_DIR + "/classifier/saved_model/dl_model.pkl")
            elif self.classifier_name == "gin":
                self.classifier = self.load_model(ROOT_DIR + "/classifier/saved_model/gin_model.pkl")
            elif self.classifier_name == "sage":
                self.classifier = self.load_model(ROOT_DIR + "/classifier/saved_model/sage_model.pkl")
            elif self.classifier_name == "ginjk":
                self.classifier = self.load_model(ROOT_DIR + "/classifier/saved_model/ginjk_model.pkl")
            else:
                self.log.info("Error: Unrecognize classifer (gspan|inria|wl|dl)")
                exit(-1)   
            self.classifier.families = families

    def init(self,exp_dir=None):
        # TODO args.binaries vs binary
        if self.input_path is None and exp_dir is None:
            self.input_path = ROOT_DIR.replace("ToolChainClassifier","output/save-SCDG") # todo add args
        elif self.input_path is None:
            self.input_path = exp_dir
        self.input_path = self.input_path.replace("unknown/","") # todo

        if self.args.families: # TODO
            self.init_classifer(args=self.args,families=self.args.families ,from_saved_model=(not self.args.train))
        else:
            families = []
            last_familiy = "unknown"
            if os.path.isdir(self.input_path):
                subfolder = [os.path.join(self.input_path, f) for f in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, f))]
                self.log.info(subfolder)
                for folder in subfolder:
                    last_familiy = folder.split("/")[-1]
                    families.append(str(last_familiy))
            self.init_classifer(args=self.args,families=families,from_saved_model=(not self.args.train))
    
    def train(self):
        if self.args.train: # TODO refactor
            args_train = {}
            if self.classifier_name == "dl":
                args_train["sepoch"] = 1#self.args.sepoch
            if self.input_path is None:
                args_train["path"] = self.input_path
            else:
                args_train["path"] = self.input_path
            self.classifier.train(**args_train)
            self.save_model(self.classifier,ROOT_DIR + "/classifier/saved_model/"+ self.classifier_name +"_model.pkl")
        
            elapsed_time = time.time() - self.start_time
            self.log.info("Total training time: " + str(elapsed_time))

    def classify(self):
        self.classifier.classify(path=(None if self.args.train else self.input_path))
        #self.log.info(self.classifier.get_stat_classifier())
    def detect(self):
        self.classifier.detection(path=(None if self.args.train else self.input_path))
    def explain(self):
        self.classifier.explain(path=(None if self.args.train else self.input_path), output_path=ROOT_DIR + "/classifier/explain_output/")

def main():
    tc = ToolChainClassifier()
    args_parser = ArgumentParserClassifier(tc)
    tc.args = args_parser.parse_arguments()
    args_parser.update_tool(tc.args)
    tc.init()
    if tc.args.explain:
        tc.explain()
    elif tc.args.train:
        tc.train()
    elif tc.mode == "classification":
        tc.classify()
        tc.classifier.get_stat_classifier()
    elif tc.mode == "detection":
        tc.detect()
        tc.classifier.get_stat_classifier()
    
    elapsed_time = time.time() - tc.start_time
    tc.log.info("Total "+ tc.mode +" time: " + str(elapsed_time))


    """if tc.args.train: # TODO
        args_res = {}
        if tc.classifier_name == "gspan":
            args_res["target"] = tc.mode
        tc.log.info(tc.classifier.get_stat_classifier(**args_res))"""

if __name__ == "__main__":
    main()
