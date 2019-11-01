import json

def get_tweet_queue(jsonfile,tweetqueue):
    if not tweetqueue.empty():
        raise Exception("the tweetqueue should be empty, you are probably loading multiple json parser processes")

    with open(jsonfile, "r", encoding="utf-8") as file:
        for a in range(100000):
            permalink = json.loads(file.readline())['permalink']
            tweetqueue.put(permalink)
