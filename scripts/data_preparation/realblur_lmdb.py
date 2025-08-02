from basicsr.utils.create_lmdb import create_lmdb_for_realblur

print("Start creating lmdb for RealBlur_J...")
create_lmdb_for_realblur(dataset_name="RealBlur_J")
#print("Start creating lmdb for RealBlur_R...")
#create_lmdb_for_realblur(dataset_name='RealBlur_R')