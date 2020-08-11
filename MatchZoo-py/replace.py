import os

n = 8
path = '/home/zhaolin/matchzoo/MatchZoo-py/event/'
for i in range(1, n+1):
    name_csv = path + 'event_test_' + str(i) + '.csv'
    new_name_csv = path + 'event_test.csv'
    os.rename(name_csv, new_name_csv)

    name_ref = path + 'event_test_' + str(i) + '.ref'
    new_name_ref = path + 'event_test.ref'
    os.rename(name_ref, new_name_ref)

    name_fil_ref = path + 'event_test_filtered_' + str(i) + '.ref'
    new_name_fil_ref = path + 'event_test_filtered.ref'
    os.rename(name_fil_ref, new_name_fil_ref)

    # 调用另外一个py文件
    os.system("python /home/zhaolin/matchzoo/MatchZoo-py/cdssm.py")
