import json

print('loading json file...')

with open('data.json') as data_file:
    data = json.load(data_file)

for i in xrange(0, 121512):
    print i
    data[i]['question_id'] = int(data[i]['question_id'])

dd = json.dump(data,open('OpenEnded_mscoco_lstm_results.json','w'))
