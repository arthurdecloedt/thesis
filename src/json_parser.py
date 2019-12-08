import json

def get_tweet_queue(jsonfile,tweetqueue):
    if not tweetqueue.empty():
        raise Exception("the tweetqueue should be empty, you are probably loading multiple json parser processes")

    with open(jsonfile, "r", encoding="utf-8") as file:
        n=0
        while True:
            line = ''
            for a in range(14):
                line= line +  file.readline().strip('\n').strip().replace(' ','')
            print(line)    
            if not line:
                break
            # if n<450000:
            #     n+=1
            #     continue
            permalink = json.loads(line)['permalink']
            tweetqueue.put(permalink)
