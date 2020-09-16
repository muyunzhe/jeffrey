#encoding: utf-8

import sys
import json
import re
reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    for line in sys.stdin:
        contents_all = []
        img_num = 0
        arr = line.strip().split('\t')
        try:
            nid, mthid, news_type, title, feed_content, pubt_time = arr
            feed_content = json.loads(feed_content)
            items = feed_content.get('items', [])
        except Exception as e:
            continue
        for item in items:
            content_type = item.get('type', '')
            if content_type == 'image':
                img_num += 1
            if content_type != 'text':
                continue
            content = item.get('data', '')
            if content:
                content = re.sub("<span class=\"[0-9a-zA-Z=-]+\">", "", content)
                content = re.sub("</span>", "", content)
                content = content.replace('\n', '')
                contents_all.append(content)
        print '\t'.join([nid, mthid, title, ''.join(contents_all), str(img_num), pubt_time])

if __name__ == "__main__":
    main()
