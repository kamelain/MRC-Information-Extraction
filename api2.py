 # -*- coding: utf-8 -*-
from bert import QA
import sys
import codecs
 
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

model = QA("model")

if __name__ == "__main__":    
    doc = "豬大哥是個很懶散的長子，牠只用一堆茅草來蓋房子，蓋好後就在房子裡呼呼大睡。豬二哥是個貪吃的次子，牠只用釘子和木頭蓋了一座木屋，豬小弟是個勤奮又聰明的么子，想要一間安全又堅固的房子，於是蓋了一間磚屋，花了很久的時間才把房子給蓋好。"
    q = '誰用茅草蓋房子？'


    f = open('result.txt', encoding='utf-8', mode="w")

    out = model.predict(doc,q)
    print(type(out))
    print(out.keys())
    # print(out['answer'].encode(encoding="utf-8"))
    ans = str(out['answer'].encode(encoding="utf-8"), 'utf-8')
    print(ans)
    f.write(ans)
