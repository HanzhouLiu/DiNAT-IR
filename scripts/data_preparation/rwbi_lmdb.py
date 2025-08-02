from basicsr.utils.create_lmdb import create_lmdb_for_rwbi

#print("Start creating lmdb for RealBlur_J...")
#create_lmdb_for_realblur(dataset_name="RealBlur_J")
print("Start creating lmdb for RWBI...")
create_lmdb_for_rwbi(dataset_name='RWBI')