"""Optional delegate to the user's ProG checkout; never invent a new SSL objective."""
import argparse,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prog-root',required=True,help='Existing ProG checkout containing pre_train.py')
    p.add_argument('--dataset',required=True)
    p.add_argument('--method',choices=['GraphMAE','Edgepred_Gprompt'],default='GraphMAE')
    p.add_argument('--epochs',type=int,required=True,help='Declare the pretraining budget explicitly')
    p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();root=Path(args.prog_root).resolve()
    if not (root/'pre_train.py').is_file():raise FileNotFoundError(root/'pre_train.py')
    subprocess.run([sys.executable,str(root/'pre_train.py'),'--dataset_name',args.dataset,
                    '--pretrain_task',args.method,'--gnn_type','GCN','--num_layer','2',
                    '--hid_dim','128','--epochs',str(args.epochs),'--device',args.device,
                    '--seed',str(args.seed)],cwd=root,check=True)

if __name__=='__main__':main()
