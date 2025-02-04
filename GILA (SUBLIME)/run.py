import sys
import pandas as pd
import numpy as np
import random
import subprocess

#SUBLIME

imb_ratio_list = [0.2]


for imb_ratio in imb_ratio_list:
    subprocess.run([sys.executable, 'main.py',
    '-type_learner','att',
    '-dataset','cora',
    '-imb_ratio',f"{imb_ratio}",
    ])

for imb_ratio in imb_ratio_list:
    subprocess.run([sys.executable, 'main.py',
    '-dataset','citeseer',
    '-type_learner','att',
    '-epochs', '1000',
    '-lr','0.001',
    '-w_decay_cls','0.05',
    '-k','20',
    '-dropedge_rate','0.25',
    '-dropedge_cls','0.5',
    '-maskfeat_rate_anchor','0.8',
    '-maskfeat_rate_learner','0.6',
    '-activation_learner','tanh',
    '-tau','0.999',
    '-imb_ratio',f"{imb_ratio}",
    ])

for imb_ratio in imb_ratio_list:
    subprocess.run([sys.executable, 'main.py',
    '-dataset','pubmed',
    '-type_learner','att',
    '-epochs', '1500',
    '-lr','0.001',
    '-lr_cls','0.01',
    '-hidden_dim','128',
    '-rep_dim','64',
    '-proj_dim','64',
    '-k','10',
    '-c','50',
    '-maskfeat_rate_learner','0.4',
    '-maskfeat_rate_learner','0.4',
    '-contrast_batch_size','2000',
    '-imb_ratio',f"{imb_ratio}"
    ])


# Imb_GSL(SUBLIME)

for imb_ratio in imb_ratio_list:
    subprocess.run([sys.executable, 'main.py',
    '-type_learner','att',
    '-dataset','cora',
    '-imb_ratio',f"{imb_ratio}",
    '-fake_samples', 'True',
    '-updaters','True',
    '-mix_node_init','True',

    ])

for imb_ratio in imb_ratio_list:
    subprocess.run([sys.executable, 'main.py',
    '-dataset','citeseer',
    '-type_learner','att',
    '-epochs', '1000',
    '-lr','0.001',
    '-w_decay_cls','0.05',
    '-k','20',
    '-dropedge_rate','0.25',
    '-dropedge_cls','0.5',
    '-maskfeat_rate_anchor','0.8',
    '-maskfeat_rate_learner','0.6',
    '-activation_learner','tanh',
    '-tau','0.999',
    '-imb_ratio',f"{imb_ratio}",
    '-fake_samples', 'True',
    '-updaters','True',
    '-mix_node_init','True',
    
    ])

for imb_ratio in imb_ratio_list:
    subprocess.run([sys.executable, 'main.py',
    '-dataset','pubmed',
    '-type_learner','att',
    '-epochs', '1500',
    '-lr','0.001',
    '-lr_cls','0.01',
    '-hidden_dim','128',
    '-rep_dim','64',
    '-proj_dim','64',
    '-k','10',
    '-c','50',
    '-maskfeat_rate_learner','0.4',
    '-maskfeat_rate_learner','0.4',
    '-contrast_batch_size','2000',
    '-imb_ratio',f"{imb_ratio}",
    '-fake_samples', 'True',
    '-updaters','True',
    '-mix_node_init','True',

        ])


        
    
    