import json

def get_tweet_queue(jsonfile,tweetqueue):
    if not tweetqueue.empty():
        raise Exception("the tweetqueue should be empty, you are probably loading multiple json parser processes")

    with open(jsonfile, "r", encoding="utf-8") as file:
        n=0
        while True:
            line = ''
            for a in range(13):
                line= file.readline()
            if not line:
                break
            line = line.strip('\n').strip()
            url = (line.split(':')[1] + ':' +line.split(':')[2]).strip().strip("\"")
            file.readline()
            # if n<450000:
            #     n+=1
            #     continue
            # permalink = json.loads(line)['permalink']
            tweetqueue.put(url)
