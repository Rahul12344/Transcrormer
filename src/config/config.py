import os
import yaml

def DNABERTCONFIG():
    return 'https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json'

def config():
        # load config from config.yaml
    with open('src/config/config.yaml', 'r') as f:
        c = yaml.safe_load(f)
    # init with config from utils/config.yaml
    #wandb.init(project="dnabert-srna", config=c)
    
    # update data directory based on whether running on aws
    env_usr = os.environ.get("USER")
    c['aws']= True if "ubuntu" in env_usr else False
    c['data-directory']=c['data-directory-ec2'] if c['aws'] else c['data-directory-local']
    
    # for every key in config that ends with 'dir', add the data directory to the beginning
    for key in c:
        if key.endswith('dir'):
            c[key] = os.path.join(c['data-directory'], c[key])
            os.makedirs(c[key], exist_ok=True)
            
    if c['sequence-mean']:
        os.makedirs(os.path.join(c['embedding_dir'], 'sequence_mean'), exist_ok=True)
        c['embedding_dir'] = os.path.join(c['embedding_dir'], 'sequence_mean')
    else:
        os.makedirs(os.path.join(c['embedding_dir'], 'sequence_true'), exist_ok=True)
        c['embedding_dir'] = os.path.join(c['embedding_dir'], 'sequence_true')
    
    c['files']={    
        'gtex_expression_values':         os.path.join(c['data-directory'], 'GSE219045_Mouse_FPKM_table_Mar22.csv'), # dataset of sRNA from BSRD
        'transcript_fasta':               os.path.join(c['data-directory'], 'HostDB-63_MmusculusC57BL6J_AnnotatedTranscripts.fasta'), # fasta file of mouse transcriptome
    }    
    
    os.makedirs(os.path.join(os.path.abspath(os.getcwd()), "model"), exist_ok=True) 
    c['model'] = os.path.join(os.path.abspath(os.getcwd()), "model")
    

    return c