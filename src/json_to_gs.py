import json
import argparse
MAPPING = {}
from link_bwt_call import *


#Create a mapping for the different syscall name and an unique identifier.
#args : mapping = name of the file for the mapping to use (format : id syscallname\n)
def create_mapping(mapping_file):
    map_file = open(mapping_file,'r')
    for line in map_file:
         tab = line.split('\n')[0].split(' ')
         MAPPING[tab[1]] = tab[0]
    map_file.close()
         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonfile", help="Name of the jsonfile to convert")
    parser.add_argument("--outfile", help="Name of the jsonfile after conversion")
    args = parser.parse_args()

    isTxt = ('.txt' in args.jsonfile)
    

    rawfile = open(args.jsonfile,'r')
    
    if not isTxt:
        jfile = json.load(rawfile)
    

    nameFile = args.jsonfile.split('.')[0].split('/')[-1]
    


            
    if isTxt or 'nodes' not in jfile:
        SCDG_FIN = []
        if isTxt:
            for line in rawfile:
                #print(line)
                a = line.replace('\n','')
                b = a.replace('\t','')
                c = b.strip()
                #d = c.replace("<","'<").replace(">",">'").replace("'<=","<=").replace(">' ","> ")
                SCDG_FIN.append(eval(c))
        
        else:
            for k in jfile.keys():
                if k != 'sections':
                    SCDG_FIN.append(jfile[k]['trace'])
        g = Graph_Builder(name=nameFile,mapping='mapping.txt',merge_call=True,comp_args=True,min_size=5,ignore_zero=True,odir='gs',verbose=True)
        
        g.build_graph(SCDG_FIN,format_out='gs')
      
    else:              
        create_mapping('mapping.txt')

        if  args.outfile:
            outfile = open(args.outfile,'w')
        else :
            outfile = open('gs/'+nameFile+'.gs','w')
        outfile.write('t # 0\n')

        
        nodes = jfile['nodes']
        
        for node in nodes:
            label = MAPPING[node['name']]
            outfile.write('v '+str(node['id'])+' '+str(label)+'\n')
            
        links = jfile['links']
        
        for edge in links:
            label = edge['label']
            outfile.write('e '+str(edge['id1'])+' '+str(edge['id2'])+' '+str(label)+'\n')      
              
        outfile.close()

    rawfile.close()
    

