# import pandas as pd
#
# train, valid, test = pd.read_csv('train.txt', header = None, sep='\t'), pd.read_csv('valid.txt', header = None, sep='\t'), pd.read_csv('test.txt', header = None, sep='\t')
#
# train_head, train_tail = set(list(train[0])), set(list(train[2]))
# valid_head, valid_tail = set(list(valid[0])), set(list(valid[2]))
# test_head, test_tail = set(list(test[0])), set(list(test[2]))
# train_ent = train_head.union(train_tail)
# valid_ent = valid_head.union(valid_tail)
# test_ent = test_head.union(test_tail)
# ent = train_ent | valid_ent | test_ent
# print(ent)
# print(len(ent))
#
# exit(0)
# train_head, train_tail = set(list(train[0])), set(list(train[-1]))
# print(train_head, train_tail)
# exit(0)
# print(train_head, train_tail)
#
