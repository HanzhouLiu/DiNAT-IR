import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#model_name = 'DeblurDiNATNoChan-GoPro-width16'
#test_layer = Net(dim=16, chan_adapt=False, mode='hybrid').cuda()  # Ablation study settings

#model_name = 'DeblurDiNATChan-GoPro-width16'
#test_layer = Net(dim=16, chan_adapt=True, mode='hybrid').cuda()  # Ablation study settings

#model_name = 'DeblurDiNATNoChanLocal-GoPro-width16'
#test_layer = Net(dim=16, chan_adapt=False, mode='local').cuda()  # Ablation study settings

model_name = 'DeblurDiNATChanLocal-GoPro-width16'
#test_layer = Net(dim=16, chan_adapt=True, mode='local').cuda()  # Ablation study settings
    
#model_name = 'DeblurDiNATNoChanGlobl-GoPro-width16'
#test_layer = Net(dim=16, chan_adapt=False, mode='globl').cuda()  # Ablation study settings

#model_name = 'DeblurDiNATChanGlobl-GoPro-width16'
#test_layer = Net(dim=16, chan_adapt=True, mode='globl').cuda()  # Ablation study settings

# Load precomputed ERF
out_dir = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/erf'
erf = np.load(os.path.join(out_dir, f'erf_avg_{model_name}.npy'))


# Plot
"""
sns_plot = plt.figure(figsize=(10, 8))
sns.heatmap(erf, cmap='Blues_r', linewidths=0.0, vmin=0, vmax=0.2,
            xticklabels=False, yticklabels=False, cbar=True)

# Save figure
out_way = os.path.join(out_dir, model_name + '_heatmap_br_nolog.png')
sns_plot.savefig(out_way, dpi=700)

"""
# Log scaling (optional but helps for subtle differences)
erf_log = np.log(erf + 1)
#erf_log -= erf_log.min()
erf_log /= erf_log.max()
erf_log = np.clip(erf_log, 0, 0.02)
erf_log /= np.max(erf_log)

# Plot with stronger color contrast
sns.set(style='white')
fig = plt.figure(figsize=(8, 8))
ax = sns.heatmap(erf_log,
                    xticklabels=False,
                    yticklabels=False, cmap="Paired", # Paired
                    center=0, annot=False, ax=None, cbar=True, square=True, annot_kws={"size": 24}, fmt='.2f')

# Save figure
out_path = os.path.join(out_dir, model_name + '_heatmap_log_mako.png')
fig.savefig(out_path, dpi=700, bbox_inches='tight')
#"""