from cleanfid import fid
import random

# https://pypi.org/project/clean-fid/
# pip install clearn-fid

#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATChan-GoPro-width16/visualization/gopro-test/output'  # 9.531946187777976
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATChanGlobl-GoPro-width16/visualization/gopro-test/output'  # 10.141842856914536
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATChanLocal-GoPro-width16/visualization/gopro-test/output'  # 10.856212882696752
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATNoChan-GoPro-width16/visualization/gopro-test/output'  # 10.078158932414226
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATNoChanGlobl-GoPro-width16/visualization/gopro-test/output'  # 10.173974978782837
fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATNoChanLocal-GoPro-width16/visualization/gopro-test/output'  # 11.122093829969344
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/Baseline-GoPro-width16/visualization/gopro-test/output'  # 15.155282205102537

#fdir1 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GoPro_/test/testB'
#fdir1 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/HIDE/test/testA'

fdir2 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/Baseline-GoPro-width16/visualization/gopro-test/gt'
#fdir2 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GoPro/test/testB'
#fdir2 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GoPro_/test/testB'

#fdir2 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/HIDE/test/testB'

#fdir2 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/RealBlur_R/test/testB'
#fdir2 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/RealBlur_J/test/testB'

random.seed(0)

# clean mode, used in our experiments
score = fid.compute_fid(fdir1, fdir2)
#score = fid.compute_kid(fdir1, fdir2)

"""
# legacy mode, not used in our experiments
#score = fid.compute_fid(fdir1, fdir2, mode="legacy_pytorch")
#score = fid.compute_kid(fdir1, fdir2, mode="legacy_pytorch")
"""

print(score)