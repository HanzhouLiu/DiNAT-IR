from basicsr.utils.create_lmdb import create_lmdb_for_rain13k, create_lmdb_for_rain13k_test

#print("Start creating lmdb for Rain100H...")
#create_lmdb_for_rain13k_test(dataset_name="Rain100H")
#print("Start creating lmdb for Rain100L...")
#create_lmdb_for_rain13k_test(dataset_name='Rain100L')
#print("Start creating lmdb for Test100...")
#create_lmdb_for_rain13k_test(dataset_name='Test100')
#print("Start creating lmdb for Test1200...")
#create_lmdb_for_rain13k_test(dataset_name='Test1200')
#print("Start creating lmdb for Test2800...")
#create_lmdb_for_rain13k_test(dataset_name='Test2800')

print("Start creating lmdb for Rain13K training set...")
create_lmdb_for_rain13k()